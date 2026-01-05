import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import pandas as pd
import numpy as np
import obonet
import argparse
from tqdm import tqdm

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================
class StructureGAT(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
    def forward(self, x, adj):
        B, L, D = x.shape
        h = self.W(x).view(B, L, self.heads, self.head_dim)
        adj_exp = adj.unsqueeze(1) 
        out = torch.matmul(adj_exp, h.transpose(1, 2)).transpose(1, 2).reshape(B, L, D)
        return F.gelu(out)

class GraphLabelEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_indices, adj_values, num_nodes, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 注册 buffer
        self.register_buffer("adj_indices", adj_indices)
        self.register_buffer("adj_values", adj_values)
        self.num_nodes = num_nodes
        
    def forward(self, x):
        N = self.num_nodes
        adj = torch.sparse_coo_tensor(self.adj_indices, self.adj_values, (N, N)).to(x.device)
        h = x
        for layer in self.layers:
            h = layer(h)
            with torch.cuda.amp.autocast(enabled=False):
                adj_fp32 = adj.float()
                h_fp32 = h.float()
                h_out = torch.sparse.mm(adj_fp32, h_fp32)
                h = h_out.type_as(h)
            h = self.act(h)
        h = self.norm(h + x[:, :h.shape[1]]) 
        return h

class TopoGALAv3(nn.Module):
    def __init__(self, config, alphabet, go_embeddings_dummy, go_graph_dummy):
        super().__init__()
        self.esm, _ = esm.pretrained.load_model_and_alphabet(config.ESM_MODEL_PATH)
        self.esm_dim = self.esm.embed_dim 
        self.hidden_dim = 512
        
        self.struct_gat = StructureGAT(self.esm_dim, heads=4)
        self.seq_proj = nn.Linear(self.esm_dim, self.hidden_dim)
        self.norm_seq = nn.LayerNorm(self.hidden_dim)
        
        # 使用传入的 dummy 数据初始化结构，但尺寸必须与 checkpoint 一致
        self.label_encoder = GraphLabelEncoder(
            input_dim=768, 
            hidden_dim=self.hidden_dim,
            adj_indices=go_graph_dummy['indices'],
            adj_values=go_graph_dummy['values'],
            num_nodes=go_graph_dummy['num_nodes']
        )
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.output_gate = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())
        
        self.register_buffer("go_embeddings", go_embeddings_dummy)
        
    def forward(self, input_ids, adj):
        # 1. ESM & Structure
        res = self.esm(input_ids, repr_layers=[33], return_contacts=False)
        esm_feats = res["representations"][33]
        struct_feats = self.struct_gat(esm_feats, adj)
        prot_feats = self.norm_seq(self.seq_proj(esm_feats + struct_feats)) 
        
        # 2. Label Graph
        label_feats = self.label_encoder(self.go_embeddings)
        
        # 3. Cross Attention
        B = prot_feats.shape[0]
        queries = label_feats.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=queries, key=prot_feats, value=prot_feats)
        
        # 4. Output
        interaction = attn_out * queries
        gate = self.output_gate(interaction)
        logits = (interaction * gate).sum(dim=-1)
        return logits

# ==========================================
# 2. 预测器类 (Predictor - 关键修改部分)
# ==========================================
class TopoGALAPredictor:
    def __init__(self, model_path, go_id_list_path, obo_path=None, device="cuda"):
        self.device = device
        # 临时 Config
        self.config = type('Config', (), {'ESM_MODEL_PATH': "/public/home/2313041002/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"})()
        
        print("1. Loading GO resources...")
        with open(go_id_list_path, 'r') as f:
            self.go_ids = [l.strip() for l in f if l.strip()]
        self.num_classes = len(self.go_ids)
        
        self.go_names = {}
        if obo_path and os.path.exists(obo_path):
            print("   Parsing OBO file...")
            graph = obonet.read_obo(obo_path)
            for go_id in self.go_ids:
                if go_id in graph:
                    self.go_names[go_id] = graph.nodes[go_id].get('name', 'Unknown')
        
        # =========================================================
        # [修改核心] 先读取权重文件，探测图的大小，而不是瞎猜
        # =========================================================
        print(f"2. Inspecting checkpoint: {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理可能的 'module.' 前缀
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # 动态获取 adj_indices 的形状
        # 这里的 key 必须与模型定义中的名称一致
        adj_key = "label_encoder.adj_indices"
        if adj_key in state_dict:
            saved_shape = state_dict[adj_key].shape # 应该是 [2, 58599]
            num_edges = saved_shape[1]
            print(f"   Detected graph edges from checkpoint: {num_edges}")
        else:
            print("   Warning: Could not find graph structure in checkpoint, using default.")
            num_edges = 10 

        # 使用探测到的尺寸创建 Dummy 数据
        dummy_indices = torch.zeros((2, num_edges), dtype=torch.long)
        dummy_values = torch.zeros(num_edges)
        dummy_go_embs = torch.zeros((self.num_classes, 768)) # GO embedding 占位符
        
        dummy_graph = {
            'indices': dummy_indices,
            'values': dummy_values,
            'num_nodes': self.num_classes
        }
        
        # =========================================================
        # 3. 初始化模型并加载权重
        # =========================================================
        print("3. Initializing Model & Loading ESM...")
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.config.ESM_MODEL_PATH)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        self.model = TopoGALAv3(self.config, self.alphabet, dummy_go_embs, dummy_graph)
        
        print("   Loading state_dict...")
        # 此时 dummy_indices 的形状与 state_dict 中的形状完全一致，不会报错
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")

    def _generate_contact_map(self, tokens):
        """
        [Fix] 修复维度不匹配问题：
        ESM输出的contacts不含特殊token(L)，而features含特殊token(L+2)。
        需要对 contact map 进行 Padding 以对齐。
        """
        with torch.no_grad():
            results = self.model.esm(tokens, repr_layers=[33], return_contacts=True)
        
        raw_contacts = results["contacts"] # Shape: [B, L_seq, L_seq] (不含 cls/eos)
        B, L_seq, _ = raw_contacts.shape
        L_total = tokens.shape[1]          # Shape: L_seq + 2 (含 cls/eos)
        
        batch_adj = []
        for i in range(B):
            cm = raw_contacts[i] # [L_seq, L_seq]
            
            # 1. 创建全零矩阵，尺寸适配 token 长度 [L+2, L+2]
            padded_cm = torch.zeros((L_total, L_total), device=tokens.device)
            
            # 2. 将真实的 Contact Map 填入中心位置 [1:-1, 1:-1]
            # 对应 tokens 中的实际氨基酸部分
            # 使用切片赋值以防万一尺寸有微小差异
            h_valid = min(L_seq, L_total - 2)
            w_valid = min(L_seq, L_total - 2)
            
            if h_valid > 0 and w_valid > 0:
                padded_cm[1:1+h_valid, 1:1+w_valid] = cm[:h_valid, :w_valid]
            
            # 3. 处理特殊 Token 的连接 (模拟训练时的逻辑)
            # 让 <cls> (索引0) 与所有节点相连，作为全局信息的汇聚点
            padded_cm[0, :] = 1.0 
            padded_cm[:, 0] = 1.0
            
            # 也可以处理 <eos> (索引-1)，通常对角线设为1即可
            padded_cm.fill_diagonal_(1.0)
            
            batch_adj.append(padded_cm)
            
        return torch.stack(batch_adj)

    def predict(self, sequences, batch_size=2, threshold=0.3, top_k=20):
        all_results = []
        
        # 分批处理
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids = [b[0] for b in batch_seqs]
            
            # 1. Tokenization
            labels, strs, tokens = self.batch_converter(batch_seqs)
            tokens = tokens.to(self.device)
            # 长序列截断 (保持与训练一致 512 + 2)
            max_len = 512 + 2
            if tokens.shape[1] > max_len:
                tokens = tokens[:, :max_len]
            
            # 2. 生成 Contact Map
            adj = self._generate_contact_map(tokens) # [B, L, L]
            
            # 3. 推理
            with torch.no_grad():
                logits = self.model(tokens, adj)
                probs = torch.sigmoid(logits)
            
            # 4. 解析结果
            probs = probs.cpu().numpy()
            for idx, prob_vec in enumerate(probs):
                top_indices = np.argsort(prob_vec)[::-1][:top_k]
                
                res_dict = {
                    "Protein_ID": batch_ids[idx],
                    "Predictions": []
                }
                
                for go_idx in top_indices:
                    score = prob_vec[go_idx]
                    if score < threshold: continue
                    
                    go_id = self.go_ids[go_idx]
                    go_name = self.go_names.get(go_id, "N/A")
                    res_dict["Predictions"].append({
                        "GO_ID": go_id,
                        "Name": go_name,
                        "Score": float(f"{score:.4f}")
                    })
                all_results.append(res_dict)
                
        return all_results

# ==========================================
# 3. 命令行接口
# ==========================================
def read_fasta(fasta_file):
    seqs = []
    header = None
    seq = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            seqs.append((header, "".join(seq)))
    return seqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topo-GALA Predictor")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Path to trained model")
    parser.add_argument("--go_list", type=str, default="data/unique_go_ids_filtered_expanded.txt", help="Path to GO ID list used in training")
    parser.add_argument("--obo", type=str, default="data/go.obo", help="Path to go.obo file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold")
    parser.add_argument("--batch_size", type=int, default=2, help="Inference batch size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found at {args.model}")
    if not os.path.exists(args.go_list):
        raise FileNotFoundError(f"GO ID list not found at {args.go_list}")
        
    predictor = TopoGALAPredictor(args.model, args.go_list, args.obo)
    
    print(f"Reading input sequences from {args.fasta}...")
    sequences = read_fasta(args.fasta)
    print(f"Found {len(sequences)} sequences.")
    
    results = predictor.predict(sequences, batch_size=args.batch_size, threshold=args.threshold)
    
    # 保存结果为 CSV
    rows = []
    for res in results:
        pid = res['Protein_ID']
        if not res['Predictions']:
            rows.append({"Protein_ID": pid, "GO_ID": None, "Name": None, "Score": None})
        for pred in res['Predictions']:
            rows.append({
                "Protein_ID": pid,
                "GO_ID": pred['GO_ID'],
                "Name": pred['Name'],
                "Score": pred['Score']
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nPrediction done! Results saved to {args.output}")
