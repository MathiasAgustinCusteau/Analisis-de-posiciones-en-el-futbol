#####################################################################
#                                                                   #
#                                                                   #
#                                                                   #
#                                                                   #
#   ESTE CODIGO NO HACE FALTA USARLO OTRA VEZ PORQUE YA ESTAN       #
#                   GENERADOS LOS ARCHIVOS                          #
#                                                                   #
#                                                                   #
#                                                                   #
#                                                                   #
#####################################################################



#%%
import os
import pandas as pd
import glob

# 1. Configurar el token e importar Kaggle
os.environ['KAGGLE_API_TOKEN'] = "KGAT_1752441f3d9dfeea2249b9684e4787a5"
import kaggle

# 2. Definir el dataset y la carpeta de destino
dataset_ref = 'emrey3lmaz/top-5-league-football-player-stats-2017-2025'
carpeta_destino = './datos_kaggle'

print(f"Descargando {dataset_ref}...")
kaggle.api.dataset_download_files(
    dataset_ref, 
    path=carpeta_destino, 
    unzip=True
)

# 3. Inspeccionar automáticamente qué archivos CSV se descargaron
archivos_csv = glob.glob(f"{carpeta_destino}/*.csv")

print("\n=== ARCHIVOS ENCONTRADOS EN EL DATASET ===")
for i, archivo in enumerate(archivos_csv):
    print(f"[{i}] {os.path.basename(archivo)}")

# 4. Cargar el archivo manejando inconsistencias en las columnas
if archivos_csv:
    archivo_a_cargar = archivos_csv[0]
    print(f"\nCargando en Pandas: {os.path.basename(archivo_a_cargar)}")
    
    try:
        # Usamos el motor de Python que tolera comas dentro de textos
        df_kaggle = pd.read_csv(
            archivo_a_cargar,
            sep=';', 
            engine='python', 
            on_bad_lines='skip'
        )
        print(f"¡Carga exitosa! Dimensiones de la matriz: {df_kaggle.shape}")
        print(df_kaggle.head())
    except Exception as e:
        print(f"Error persistente al leer el CSV: {e}")
else:
    print("No se encontraron archivos .csv en la carpeta.")

#%%

# 1. Limpiamos los nombres de las columnas por seguridad
df_kaggle.columns = df_kaggle.columns.str.strip()

# 2. Creamos una carpeta dedicada para no ensuciar la raíz de tu proyecto
carpeta_salida = './datos_kaggle/ligas_temporadas'
os.makedirs(carpeta_salida, exist_ok=True)

# 3. Agrupación bidimensional (Liga y Temporada)
# Esto crea una matriz interna separada por cada combinación única que exista
grupos = df_kaggle.groupby(['league', 'season'])

print("Generando archivos separados por Liga y Temporada...\n")

for (liga, temporada), df_grupo in grupos:
    # Sanitizamos los nombres para que los archivos no tengan caracteres inválidos en Windows
    liga_segura = str(liga).replace(' ', '_').replace('/', '-')
    temp_segura = str(temporada).strip()
    
    # Formato de salida: ej. "Premier_League_1718.csv"
    nombre_archivo = f"{liga_segura}_{temp_segura}.csv"
    ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)
    
    # Exportamos el sub-dataframe de ese torneo particular
    df_grupo.to_csv(ruta_archivo, index=False)
    print(f" -> Guardado: {nombre_archivo} (Filas: {df_grupo.shape[0]})")

print(f"\n¡Proceso finalizado! Matrices organizadas en: {carpeta_salida}")
# %%
