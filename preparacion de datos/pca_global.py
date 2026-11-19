#%%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%%
# 1. Configuración de rutas ancladas a la raíz del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
ruta_entrada = RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv'
carpeta_clustering = RAIZ_PROYECTO / 'Clustering'
carpeta_clustering.mkdir(parents=True, exist_ok=True)

df_master = pd.read_csv(ruta_entrada)

# Restauramos el índice con la metadata
metadata = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']
df_master = df_master.set_index(metadata)

# Verificamos que no haya arqueros ni NaNs
assert not df_master.index.get_level_values('pos_').astype(str).str.contains('GK').any(), "¡Ojo! Todavía hay arqueros en el dataset."
X = df_master.values 
print(f"Forma de la matriz de entrada al PCA (sin GK): {X.shape}")

# 2. Entrenamos el PCA Global exploratorio
pca = PCA()
pca.fit(X)

varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# 3. Gráfico del Codo (Scree Plot)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--')
plt.axhline(y=0.80, color='r', linestyle='-', label='80% Varianza')
plt.axhline(y=0.90, color='g', linestyle='-', label='90% Varianza')
plt.title('Varianza Explicada Acumulada por el PCA Global (Sin Arqueros)')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

comp_80 = int(np.argmax(varianza_acumulada >= 0.80) + 1)
comp_90 = int(np.argmax(varianza_acumulada >= 0.90) + 1)

print(f"Para retener el 80% de la varianza necesitás: {comp_80} componentes.")
print(f"Para retener el 90% de la varianza necesitás: {comp_90} componentes.")

#%%
# 4. Entrenamos el modelo definitivo usando el número exacto para el 90% de varianza
N_COMP = comp_90  # O fijalo manualmente (ej. 15 o 16) luego de ver el gráfico
pca_final = PCA(n_components=N_COMP)
X_pca = pca_final.fit_transform(X)

loadings = pd.DataFrame(
    pca_final.components_,
    columns=df_master.columns,
    index=[f'PC{i+1}' for i in range(N_COMP)]
)

def analizar_componente(pc_name, top_n=5):
    pesos = loadings.loc[pc_name].sort_values(ascending=False)
    idx_pc = int(pc_name[2:]) - 1
    print(f"\n{'='*40}")
    print(f" ANÁLISIS DEL COMPONENTE: {pc_name} ({pca_final.explained_variance_ratio_[idx_pc]*100:.2f}% var)")
    print(f"{'='*40}")
    print("[+] Semieje POSITIVO:")
    print(pesos.head(top_n))
    print("\n[-] Semieje NEGATIVO:")
    print(pesos.tail(top_n))

# Revisá esta salida para verificar si los nombres de tus componentes siguen coincidiendo
for i in range(1, N_COMP + 1):
    analizar_componente(f'PC{i}')

#%%
# 5. Exportación a la carpeta Clustering
# Si cambiaste N_COMP o el orden de los loadings, ajustá esta lista antes de correr esta celda
columnas_pca = [f'PC{i+1}' for i in range(N_COMP)] # Podés reemplazar por nombres_componentes cuando los verifiques

df_pca = pd.DataFrame(X_pca, columns=columnas_pca, index=df_master.index)

# Estos nombres tienen que coincidir con los que lee Clustering/clustering
ruta_salida_pca = carpeta_clustering / 'datos_pca.parquet'
ruta_salida_loadings = carpeta_clustering / 'loadings_pca.csv'

df_pca.to_parquet(ruta_salida_pca)
loadings.to_csv(ruta_salida_loadings)

print(f"\nGuardado exitoso en: {ruta_salida_pca}")
print(f"Dimensiones finales exportadas: {df_pca.shape}")
# %%
