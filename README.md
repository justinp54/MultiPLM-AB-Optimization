# Protein Language Model Ensemble & Affinity Prediction

This repository integrates:
- **ESM (Evolutionary Scale Modeling)** protein language models (PLMs) to propose and analyze mutations via ensemble voting.
- **AbAgNet ΔΔG predictor** for evaluating the effect of mutations on antibody–antigen binding affinity.
- Analysis pipelines for both **single user-provided sequences** and **therapeutic antibody datasets (Thera-SAbDab)**.

---

## 📦 Project Structure

```
.
├── source
│   ├── models.py          # Wrapper for ESM models (Model class)
│   ├── ensemble.py        # Functions for model ensemble & mutation voting
│   ├── utils.py           # Utility functions (seed_everything, one_hot, save_plot, etc.)
│   ├── ddg_predictor.py   # AbAgNet model definition for ΔΔG prediction
│   ├── custom_loader.py   # PyTorch Dataset for Ab/Ag/mutant sequences
│   ├── run.py             # CLI: ensemble scoring for a sequence
│   ├── run_analysis.py    # CLI: mutation voting & entropy plots for user-provided sequences
│   └── run_ddg_infer.py   # CLI: predict ΔΔG for mutations using AbAgNet
├── AffinityModel
│   └── ddg_predictor.pt   # Pretrained weights for AbAgNet (place here)
└── README.md              # Project description
```

---

## 🔧 Environment Setup

1. Python >= 3.8 recommended
2. PyTorch (GPU with CUDA support strongly recommended)
3. Install dependencies:

   ```bash
   pip install torch numpy pandas matplotlib fair-esm
   ```

4. (Optional) Set GPU device:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

---

## 📥 Pretrained ESM Models

Pretrained ESM models can be downloaded from the official Facebook Research GitHub:

- [ESM pretrained models page](https://github.com/facebookresearch/esm#available-models)

Main models used in this project:
- `esm1b_t33_650M_UR50S`
- `esm1v_t33_650M_UR90S_1` … `esm1v_t33_650M_UR90S_5`

They will be automatically downloaded and cached by `fair-esm` when first used.

---

## ▶ Usage

### 1. Ensemble scoring for a sequence
Run ESM ensemble scoring for a single protein sequence:

```bash
python -m source.run --seq "<PROTEIN_SEQUENCE>" \
    --models esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5 \
    --cuda 0 --seed 42
```

---

### 2. Mutation voting & entropy plots (user-provided sequences)

Analyze mutations suggested by multiple PLMs for one or more sequences:

```bash
# Single sequence + save entropy plot
python -m source.run_analysis \
  --seq "EVQLVESGGGLVQPGGSLRLSCAASGFTFTTYAMGWVRQAPGKGPEWVSLTSYDGSSTWYDDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARSLVPFAPLDYWGQGTLVTVSS" \
  --models esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5 \
  --cuda 0 --seed 0 \
  --save-plot --plot-prefix entropy --topk 10
```

```bash
# Multiple sequences, with custom names, save plots and CSV
python -m source.run_analysis \
  --seq SEQA SEQB SEQC \
  --name Ab1 Ab2 Ab3 \
  --models esm1b esm1v1 esm1v2 esm1v3 esm1v4 esm1v5 \
  --save-plot --ofname results.csv
```

Outputs:
- CSV (if `--ofname` given): `results.csv` with `Name, Sequence, Mutation, Number of Language Models`
- Plots (if `--save-plot` given): saved under `<project_root>/Analysis_results/`  
  (`ANALYSIS_RESULTS_DIR` env var can override the output directory)

---

### 3. ΔΔG prediction with AbAgNet

Predict binding free energy difference for antibody mutations:

```bash
python -m source.run_ddg_infer \
  --ab "EVQLVESGGGLVQPGGSLRLSCAASGFTFTTYAMGWVRQAPGKGPEWVSLTSYDGSSTWYDDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARSLVPFAPLDYWGQGTLVTVSS" \
  --ag "ANTIGENSEQUENCE" \
  --mut 30:A 57:Y \
  --cuda 0 --seed 0
```

- If `--weights` is not provided, the script will look for `AffinityModel/ddg_predictor.pt` by default.
- Mutation positions are 1-based by default (`--index-base 1`).  
  Use `--index-base 0` if you want to provide 0-based positions.

Example output:
```
30,A,-0.523114
57,Y,0.184223
```
Each line is `position,mutantAA,predicted_ddG`.

Optional: save results to CSV:
```bash
python -m source.run_ddg_infer \
  --ab "..." --ag "..." \
  --mut 30:A 57:Y \
  --out ddg_predictions.csv
```

---

## 📌 References

- Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*, Science (2021)  
- GitHub: [facebookresearch/esm](https://github.com/facebookresearch/esm)
