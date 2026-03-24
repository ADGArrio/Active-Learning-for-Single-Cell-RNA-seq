# %%
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# %%
adata = sc.read_h5ad("Data/03390dd0-fe16-4cef-b430-ab451e85c448.h5ad")

# %% Choosing only highly variable genes
adata = adata[:, adata.var["highly_variable"]]

# %% Log-transforming and scaling the data
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# %% 
print(adata.X.shape)
print(adata.obs["cell_type"].nunique())

# %% Plot distribution of cell types

plt.figure(figsize=(10, 5))
cell_counts = adata.obs["cell_type"].value_counts()
sns.barplot(x=cell_counts.index, y=cell_counts.values)
plt.xticks(rotation=45, ha="right")
plt.title("Cell Type Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %% Plot UMAP colored by cell type
sc.pl.umap(adata, color="cell_type")

# %% Plot PCA (optional quick view)
sc.pl.pca(adata, color="cell_type")

# %%