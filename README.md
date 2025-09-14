# Protein Language Model Ensemble (ESM-based)

This repository provides code to load Facebook AI Research's [ESM (Evolutionary Scale Modeling)](https://github.com/facebookresearch/esm) protein language models (PLMs) 
and apply them jointly (ensemble) to compute probabilities and logits for protein sequences.

---

## 📦 Project Structure

```
.
├── source
│   ├── models.py       # Wrapper for ESM models (Model class)
│   ├── ensemble.py     # Functions for model ensemble
│   ├── utils.py        # Utility functions (compare, seed_everything)
│   └── run.py          # CLI entry point
└── README.md           # Project description
```

---

## 🔧 Environment Setup

1. Python >= 3.8 recommended
2. PyTorch (GPU with CUDA support strongly recommended)
3. Install dependencies:

   ```bash
   pip install torch numpy fair-esm
   ```

4. (Optional) Set GPU device:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

---

## 📥 Pretrained Models

Pretrained ESM models can be downloaded from the official Facebook Research GitHub:

- [ESM pretrained models page](https://github.com/facebookresearch/esm#available-models)

Main models used in this project:
- `esm1b_t33_650M_UR50S`
- `esm1v_t33_650M_UR90S_1`
- `esm1v_t33_650M_UR90S_2`
- `esm1v_t33_650M_UR90S_3`
- `esm1v_t33_650M_UR90S_4`
- `esm1v_t33_650M_UR90S_5`

The models will be automatically downloaded and cached by the `fair-esm` library when used for the first time.

---

## ▶ Usage

The `run.py` script provides a command-line interface (CLI):

```bash
python -m source.run --seq "<PROTEIN_SEQUENCE>" \
    --models esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5 \
    --cuda 0 \
    --seed 42
```

### Arguments
- `--seq` : Input protein sequence (required)
- `--models` : List of PLM model names (default: `esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5`)
- `--cuda` : GPU device id (e.g., `"0"`). If not set, CPU is used
- `--seed` : Random seed for reproducibility

---

## 🧪 Example Run

Example using the antibody heavy chain sequence from the original notebook:

```bash
python -m source.run \
  --seq "EVQLVESGGGLVQPGGSLRLSCAASGFTFTTYAMGWVRQAPGKGPEWVSLTSYDGSSTWYDDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARSLVPFAPLDYWGQGTLVTVSS" \
  --models esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5 \
  --cuda 0 \
  --seed 0
```

### Example Output
```
models: ['esm1b', 'esm1v1', 'esm1v2', 'esm1v3', 'esm1v4', 'esm1v5']
ensemble logits: (L, A)
ensemble probs: (L, A)
```

Where:
- `L` = protein sequence length
- `A` = number of tokens (amino acids)

---

## 📌 References

- Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*, Science (2021)  
- GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
