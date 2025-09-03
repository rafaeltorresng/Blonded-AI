import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_loader import load_dataset

# Caminhos
dataset_path = "data/processed_dataset.csv"
scaler_path = "model/scaler_model.pkl"
pca_path = "model/pca_model.pkl"

# 1. Carregar dataset e features
dataset, feature_cols = load_dataset(dataset_path)
features = dataset[feature_cols]  # DataFrame, mantém nomes das colunas

# 2. Treinar e salvar o scaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler salvo em {scaler_path}")

# 3. Treinar e salvar o PCA
pca = PCA(n_components=6)
pca.fit(scaled_features)
with open(pca_path, "wb") as f:
    pickle.dump(pca, f)
print(f"PCA salvo em {pca_path}")