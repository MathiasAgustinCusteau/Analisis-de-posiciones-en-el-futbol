#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%%
# 1. Cargamos la matriz maestra que armamos en el paso anterior
carpeta_save = '../datos_kaggle/procesados'
df_master = pd.read_csv(f"{carpeta_save}/matriz_master_pca.csv")

# Restauramos el índice con la metadata porque read_csv lo carga como columnas normales
metadata = ['league', 'season', 'team', 'player', 'nation_', 'pos_']
df_master = df_master.set_index(metadata)

# Verificamos que solo queden números puros sin NaNs
X = df_master.values 
print(f"Forma de la matriz de entrada al PCA: {X.shape}")

# 2. Entrenamos el PCA Global
# Inicialmente no le limitamos los componentes para ver toda la curva de varianza
pca = PCA()
pca.fit(X)

# 3. Análisis de Varianza Explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# 4. Gráfico del Codo (Scree Plot)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--')
plt.axhline(y=0.80, color='r', linestyle='-', label='80% Varianza')
plt.axhline(y=0.90, color='g', linestyle='-', label='90% Varianza')

plt.title('Varianza Explicada Acumulada por el PCA Global')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 5. Calculamos cuántos componentes necesitamos para el 80% y 90% de la información
comp_80 = np.argmax(varianza_acumulada >= 0.80) + 1
comp_90 = np.argmax(varianza_acumulada >= 0.90) + 1

print(f"Para retener el 80% de la varianza necesitás: {comp_80} componentes.")
print(f"Para retener el 90% de la varianza necesitás: {comp_90} componentes.")

#%%
# 1. Reentrenamos el modelo fijando los 15 componentes definitivos
pca_final = PCA(n_components=15)
X_pca = pca_final.fit_transform(X) # Esta es tu nueva matriz de 12848 filas y 15 columnas

# 2. Extraemos los "loadings" (pesos)
# La matriz components_ tiene forma (n_components, n_features)
loadings = pd.DataFrame(
    pca_final.components_,
    columns=df_master.columns,
    index=[f'PC{i+1}' for i in range(15)]
)

# 3. Función para leer la física del componente
def analizar_componente(pc_name, top_n=5):
    """
    Imprime las variables que más tiran hacia el lado positivo y negativo del eje.
    """
    pesos = loadings.loc[pc_name].sort_values(ascending=False)
    
    print(f"\n{'='*40}")
    print(f" ANÁLISIS DEL COMPONENTE: {pc_name}")
    print(f" Varianza explicada: {pca_final.explained_variance_ratio_[int(pc_name[2:])-1]*100:.2f}%")
    print(f"{'='*40}")
    
    print("\n[+] Tiran hacia el semieje POSITIVO (Si el jugador tiene un valor alto acá, hace mucho de esto):")
    print(pesos.head(top_n))
    
    print("\n[-] Tiran hacia el semieje NEGATIVO (Si el jugador tiene un valor bajo/negativo acá, hace mucho de esto):")
    print(pesos.tail(top_n))
#%%
# 4. Inspeccionamos los primeros 4 componentes principales (los que más varianza explican)
for i in range(1, 16):
    analizar_componente(f'PC{i}')

#%%

# 1. Definís tu lista de 15 nombres basados en tu análisis de los pesos (loadings)
# los nombres se pusieron tomando los pesos positivos como lo principal para describir la funcion. si dice algo vs algo el primero representa peso positivo y el segundo el peso negativo
nombres_componentes = [
    'ofensividad',               # PC1: Delantero vs Defensor
    'creacion_y_distribucion',   # PC2: Armador vs finalizador
    'intercepcion_de_ataques',   # PC3: Defensor rústico vs Jugador de creacion
    'juego_aereo_ofensivo',      # PC4: El 9 clasico vs jugador en la banda/mediocampo
    'proyeccion_por_bandas',     # PC5: lateral/extremo vs mediocampista todoterreno/ofensivo
    'regate_y_desequilibrio',    # PC6: jugador desequilibrante vs jugador asistidor
    'salida_larga_y_bloqueos',   # PC7: despeje y defensa vs manejo de pelota ofensivo
    'presion_y_juego_directo',  # PC8: mantener posesion y terminar jugadas vs presion y finalizacion ofensiva
    'salida_bajo_presion',       # PC9: salida y proyeccion al ataque vs atacante creativo (parece una separacion entre lateral y extremo)
    'pivoteo_y_asistencia',      # PC10: pases arriesgados vs finalizacion
    'distribucion y recuperacion',# PC11: distribucion arriesgada vs distribucion firme (parecen ser un 5 vs un central)
    'bloqueos_y_presion_alta',   # PC12: parece lateral ofensivo vs lateral defensivo
    'filtrado_sin_asistencia',   # PC13: separa creativos sin asistencia vs creativos que terminan y asisten (parece un creativo que presiona y juega tranquilo vs uno que presiona y juega fuerte)
    'retencion_y_distribucion',  # PC14: parece distinguir entre estilos de mediocampistas (como un 5 de control vs un 5/8 de salida y contra ataque)
    'presion_alta_liviana'       # PC15: presionar fuerte vs presionar bajo
]

# 2. Convertimos el output del PCA (X_pca) en un DataFrame con tus columnas
df_pca = pd.DataFrame(X_pca, columns=nombres_componentes)

# 3. Le devolvemos el índice original (los metadatos) para saber de qué jugador es cada fila
df_pca.index = df_master.index

# Tu matriz ahora es 100% interpretable y está lista para el clustering
print(df_pca.head())
# %%
