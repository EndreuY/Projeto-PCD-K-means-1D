import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# --- Parâmetros para Geração de Dados ---
# Conjunto de dados de tamanho médio, bom para testes no Colab
N_SAMPLES = 500000  # 500 mil pontos
N_CLUSTERS = 8      # K = 8
FILE_DATA = 'dados.csv'
FILE_CENTROIDS = 'centroides_iniciais.csv'

print(f"Gerando conjunto de dados com N={N_SAMPLES} e K={N_CLUSTERS}...")

# Gera pontos 1D agrupados
X, y = make_blobs(n_samples=N_SAMPLES, centers=N_CLUSTERS, n_features=1,
                  center_box=(-100.0, 100.0), cluster_std=5.0, random_state=42)

# Salva os dados dos pontos
pd.DataFrame(X).to_csv(FILE_DATA, header=False, index=False)

# Escolhe K pontos aleatórios dos dados como centróides iniciais
initial_centroids_indices = np.random.choice(len(X), N_CLUSTERS, replace=False)
initial_centroids = X[initial_centroids_indices]

# Salva os centróides iniciais
pd.DataFrame(initial_centroids).to_csv(FILE_CENTROIDS, header=False, index=False)

print(f"Arquivos '{FILE_DATA}' e '{FILE_CENTROIDS}' gerados com sucesso.")