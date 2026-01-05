# TopoMem: Topology-Guided Prediction of Membrane Protein Functions via Weighted Contact Maps

This repo is for membrane protein function prediction research. 

## MemGO-Bench: Our own made protein dataset specially for membrane protein research

Here we introduce the files briefly: 

- membrane_proteins_original.xlsx

We collected data of a total of 47,260 membrane proteins from the PDB and UniProt databases.

![数据集构建流程](./数据集构建流程.jpg)

- membrane_proteins.xlsx

Based on the original data file, the GO annotations of the sequence are expanded through the QuickGO database.

- membrane_proteins_supplement_GOs_domain_filtered.xlsx

We removed those proteins without domains from InterPro database that left us 44539 records.

## Quick Start

### 1. Environment Setup
- Python 3.8+
- Required packages: `torch`, `esm`, `pandas`, `obonet`, `tqdm`
- GPU (CUDA) recommended for faster inference

### 2. Download Required Files
- Model weights: `checkpoints_v3/best_model_v3.pth`
- GO ID list: `data/unique_go_ids_filtered_expanded.txt`
- GO OBO file (for name resolution): `data/go.obo`

### 3. Test Samples
Save the following content as `test_samples.fasta`:

```fasta
>sp|P00760|TRY1_BOVIN Chymotrypsinogen A (Bos taurus) - Protease
CGVPAIQPVLSGLSRIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTTSDVVVAGEFDQGSSSEKIQKLKIAKVFKNSKYNSLTINNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYTNANTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGASGVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWQQTLAAN

>sp|P00698|LYC_CHICK Lysozyme C (Gallus gallus) - Antimicrobial enzyme
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

>sp|P61823|RNAS1_BOVIN Ribonuclease pancreatic (Bos taurus) - RNA degradation
KESRAKKFQRQHMDSDSSPSSSSTYCNQMMKRRKMTLYHCKRFNTFIHEDIWNIRSICSTTNIQCKNGKMNCHEGVVKVTDCRDTGSSRAPNCRYRAIASTRRVVIACEGNPQVPVHFDG

>sp|P02232|INS_HUMAN Insulin (Homo sapiens) - Hormone (proinsulin form, short)
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
```

### 4. Run Prediction
```bash
python predict.py \
  --fasta test_samples.fasta \
  --model checkpoints_v3/best_model_v3.pth \
  --go_list data/unique_go_ids_filtered_expanded.txt \
  --obo data/go.obo \
  --output predictions.csv \
  --threshold 0.3 \
  --batch_size 2
```

Inference time: a few seconds to tens of seconds (GPU-dependent).

## Example Prediction Results (Threshold = 0.3)

Below are actual predictions for the test samples (top terms, sorted by score). The model performs exceptionally well on serine proteases, aligning closely with UniProt annotations.

### 1. Bovine Chymotrypsinogen A - Serine Protease


**Highlight**: Near-perfect scores for core serine protease functions (matches UniProt P00760 exactly, e.g., GO:0004252, GO:0008236).

| Rank | GO_ID      | Name                                   | Score  |
|------|------------|----------------------------------------|--------|
| 1    | GO:0004175 | endopeptidase activity                 | 0.9998 |
| 2    | GO:0017171 | serine hydrolase activity              | 0.9995 |
| 3    | GO:0008236 | serine-type peptidase activity         | 0.9993 |
| 4    | GO:0004252 | serine-type endopeptidase activity     | 0.9992 |
| 5    | GO:0016787 | hydrolase activity                     | 0.9987 |
| 6    | GO:0008233 | peptidase activity                     | 0.9975 |
| 7    | GO:0006508 | proteolysis                            | 0.9952 |

### 2. Chicken Lysozyme C - Antimicrobial Enzyme



**Highlight**: Correctly identifies hydrolase activity and extracellular localization (reasonable), though the most specific "lysozyme activity (GO:0003796)" is not top-ranked—tends toward broader terms.

| Rank | GO_ID      | Name                                   | Score  |
|------|------------|----------------------------------------|--------|
| 1    | GO:0110165 | cellular anatomical structure          | 0.9976 |
| 2    | GO:0008150 | biological_process                     | 0.9965 |
| 3    | GO:0016787 | hydrolase activity                     | 0.8152 |
| 4    | GO:0005576 | extracellular region                   | 0.8010 |
| 5    | GO:0003824 | catalytic activity                     | 0.6642 |

### 3. Bovine Pancreatic Ribonuclease - RNA Degrading Enzyme



**Highlight**: Accurately captures defense responses and nucleic acid catalysis, consistent with the known antibacterial role of the RNase A family.

| Rank | GO_ID      | Name                                           | Score  |
|------|------------|------------------------------------------------|--------|
| 1    | GO:0050830 | defense response to Gram-positive bacterium    | 0.9966 |
| 2    | GO:0050829 | defense response to Gram-negative bacterium    | 0.9846 |
| 3    | GO:0140640 | catalytic activity, acting on a nucleic acid   | 0.9474 |
| 4    | GO:0005576 | extracellular region                           | 0.9501 |
| 5    | GO:0042742 | defense response to bacterium                  | 0.9362 |

### 4. Human Proinsulin - Hormone



**Highlight**: Strongly predicts regulation and metabolic processes, aligning with insulin's role in signaling and glucose homeostasis (UniProt P01308).

| Rank | GO_ID      | Name                                   | Score  |
|------|------------|----------------------------------------|--------|
| 1    | GO:0008150 | biological_process                     | 1.0000 |
| 2    | GO:0050794 | regulation of cellular process         | 1.0000 |
| 3    | GO:0008152 | metabolic process                      | 1.0000 |
| 4    | GO:0019222 | regulation of metabolic process        | 0.9999 |
| 5    | GO:0023052 | signaling                              | 0.9999 |

These examples demonstrate the model's generalization across protein types, with particular strength in enzymatic and regulatory functions. The full CSV output includes all predictions with score > 0.3.

Feel free to contribute, test additional proteins, or adjust the threshold to explore lower-confidence predictions!
