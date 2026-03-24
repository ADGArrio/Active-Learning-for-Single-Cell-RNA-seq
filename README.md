# Active Learning for Single-Cell RNA-seq (Mouse Brain Aging)

## 📌 Overview
This project explores **active learning with deep learning** for classifying cell types using single-cell RNA sequencing (scRNA-seq) data from mouse brain aging datasets.

We aim to go beyond classical machine learning approaches by:
- Using a **neural network (MLP)** as the base learner
- Implementing **active learning strategies**
- Studying how **uncertainty and batch size** affect performance

---

## 🧬 Dataset
We use the **Brain Aging Spatial Atlas (snRNA-seq)** dataset:
- Organism: *Mus musculus*
- Cells: ~79,000
- Genes (features): ~3,000 (after preprocessing)
- Task: **Cell type classification**

Each sample corresponds to a single cell with:
- Gene expression values (input features)
- Cell type labels (target)

---

## ⚙️ Data Preprocessing
We apply standard scRNA-seq preprocessing steps:
- Log normalization (`log1p`)
- Feature selection (highly variable genes)
- Scaling

We also visualize:
- Cell type distribution
- UMAP and PCA embeddings

---

## 🧠 Model
We use a **Multi-Layer Perceptron (MLP)**:
- Input: gene expression vector
- Hidden layers: fully connected + ReLU
- Output: softmax over cell types

---

## 🎯 Active Learning Setup
We simulate a **labeling scenario**:
- Start with a small labeled dataset
- Iteratively select new samples to label

### Query Strategies
We compare:
- Random sampling (baseline)
- Entropy-based uncertainty
- Monte Carlo Dropout (uncertainty via stochastic forward passes)

### Batch Selection
We explore:
- One-at-a-time selection
- Batch selection (varying batch sizes)

---

## 🧪 Experiments
We evaluate:
- Model accuracy vs. number of labeled samples
- Effect of:
  - Uncertainty strategy
  - Batch size
  - Initialization (random vs. DoE)

---

## 📊 Key Questions
- Does active learning outperform random sampling?
- Which uncertainty method works best?
- How does batch size affect performance?
- Can active learning better capture rare cell types?

---

## 🗂️ Project Structure
```
asr_active_learning_project/
│
├── data/              # (ignored) raw datasets
├── notebooks/         # exploratory analysis
├── src/               # model + active learning code
├── results/           # plots, outputs
├── README.md
└── .gitignore
```

---

## 🚀 Setup

### Create environment
```bash
conda create -n asr-proj python=3.10
conda activate asr-proj
```

### Install dependencies
```bash
pip install scanpy anndata numpy pandas matplotlib seaborn scikit-learn torch
```

---

## ▶️ Running the Project
1. Load and preprocess data
2. Train baseline MLP
3. Run active learning loop
4. Evaluate performance

---

## 📌 Notes
- Large datasets are not included in the repository
- Download instructions should be added separately

---

## 👥 Authors
- Arrio Gonsalves, Jeffery Yang