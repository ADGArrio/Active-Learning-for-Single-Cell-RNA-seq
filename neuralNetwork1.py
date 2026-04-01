# %% 
# import packages and libaries.

import scanpy as sc
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# %% 
# read the .h5ad file.
adata = sc.read_h5ad("Data/03390dd0-fe16-4cef-b430-ab451e85c448.h5ad")
print("Data shape:", adata.shape)

# %% 
# process the dataset.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

X = adata.X
X = X.toarray() if not isinstance(X, np.ndarray) else X
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

# %% 
# set the labels of the dataset.
y = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

# %% 
# split the dataset into training and validation sets.
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_full = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full = torch.tensor(y_train_full, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# %% 
# define the neural network model.
class RNASeqNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

    def get_embedding(self, x):
        return self.encoder(x)

# %% 
# create a diversity sampling function.
def select_diverse(model, X_labeled, X_unlabeled, k):
    model.eval()
    with torch.no_grad():
        emb_labeled = model.get_embedding(X_labeled).cpu().numpy()
        emb_unlabeled = model.get_embedding(X_unlabeled).cpu().numpy()
    emb_labeled = normalize(emb_labeled)
    emb_unlabeled = normalize(emb_unlabeled)
    distances = pairwise_distances(emb_unlabeled, emb_labeled)
    min_dist = distances.min(axis=1)
    return np.argsort(min_dist)[-k:]

# %% 
# set the parameters for active learning.
n_total = len(X_train_full)
indices = np.arange(n_total)
np.random.shuffle(indices)

initial_size = int(0.2 * n_total)
labeled_idx = indices[:initial_size]
unlabeled_idx = indices[initial_size:]

history_acc = []
history_bal_acc = []
history_f1 = []

query_size = 100
n_iterations = 10
epsilon = 0.1

# %% 
# run active learning.
for it in range(n_iterations):
    print(f"\n=== Iteration {it+1} ===")

    X_train = X_train_full[labeled_idx]
    y_train = y_train_full[labeled_idx]
    X_unlabeled = X_train_full[unlabeled_idx]

    # set the weights of each class.
    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_torch = torch.tensor(class_weights, dtype=torch.float32)

    model = RNASeqNet(input_dim=X.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights_torch)

    for epoch in range(20):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        preds = torch.argmax(val_outputs, dim=1)

    acc = accuracy_score(y_val.numpy(), preds.numpy())
    bal_acc = balanced_accuracy_score(y_val.numpy(), preds.numpy())
    f1 = f1_score(y_val.numpy(), preds.numpy(), average='macro')

    history_acc.append(acc)
    history_bal_acc.append(bal_acc)
    history_f1.append(f1)

    print(f"Acc: {acc:.4f} | Bal Acc: {bal_acc:.4f} | Macro F1: {f1:.4f}")

    print(classification_report(y_val.numpy(), preds.numpy(), zero_division=0))

    if len(unlabeled_idx) == 0:
        break

    # selecting samples to label.
    k = min(query_size, len(unlabeled_idx))
    if np.random.rand() < epsilon:
        selected = np.random.choice(len(unlabeled_idx), size=k, replace=False)
    else:
        selected = select_diverse(model, X_train, X_unlabeled, k)

    new_labeled = unlabeled_idx[selected]
    labeled_idx = np.concatenate([labeled_idx, new_labeled])
    unlabeled_idx = np.setdiff1d(unlabeled_idx, new_labeled)

    print(f"Labeled samples: {len(labeled_idx)}")

# %% 
# evaluate the accuracy of the model.
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_val), dim=1).numpy()

print("\nFinal Metrics:")
print("Accuracy:", accuracy_score(y_val.numpy(), preds))
print("Balanced Accuracy:", balanced_accuracy_score(y_val.numpy(), preds))
print("Macro F1:", f1_score(y_val.numpy(), preds, average='macro'))
print(classification_report(y_val.numpy(), preds, zero_division=0))

# %% 
# plot the results.
plt.plot(history_acc, label="Accuracy")
#plt.plot(history_bal_acc, label="Balanced Accuracy")
#plt.plot(history_f1, label="Macro F1")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Active Learning Performance")
plt.legend()
plt.grid()
plt.show()

# %%
# plot bar graph of cell type distribution.
cell_counts = adata.obs['cell_type'].value_counts()

plt.figure(figsize=(12,5))
plt.bar(cell_counts.index, cell_counts.values, color='skyblue')
plt.xticks(rotation=90)
plt.ylabel("Number of Cells")
plt.title("Cell Count per Cell Type")
plt.tight_layout()
plt.show()

# make a UMAP plot.
if 'X_umap' not in adata.obsm.keys():
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color='cell_type',
    legend_loc='right margin',
    title='Cell Type UMAP'
)

# make a heatmap of the genes.
if 'highly_variable' not in adata.var.columns:
    sc.pp.highly_variable_genes(adata, n_top_genes=50)

top_hvg = adata.var[adata.var['highly_variable']].sort_values('dispersions_norm', ascending=False).head(50).index
adata_hvg = adata[:, top_hvg]

sc.pl.heatmap(
    adata_hvg, 
    var_names=top_hvg, 
    groupby='cell_type', 
    swap_axes=True, 
    cmap='viridis', 
    show_gene_labels=True,
    figsize=(12,8)
)
