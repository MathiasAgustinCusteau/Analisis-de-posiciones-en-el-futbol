#####################################################################
#                                                                   #
#                                                                   #
#                                                                   #
#                                                                   #
#   ESTE CODIGO NO HACE FALTA USARLO OTRA VEZ PORQUE YA ESTAN       #
#                   GENERADOS LOS ARCHIVOS                          #
#                                                                   #
#          (al final se puede ver el daraframe generado)            #
#                                                                   #
#                                                                   #
#####################################################################



#%%
import pandas as pd
import numpy as np
import os
import glob

# 1. Configuración de directorios
carpeta_read = './datos_kaggle/ligas_temporadas'
carpeta_save = './datos_kaggle/procesados'

# Crear la carpeta de guardado si no existe
os.makedirs(carpeta_save, exist_ok=True)

archivos_csv = glob.glob(f"{carpeta_read}/*.csv")
print(f"=== SE ENCONTRARON {len(archivos_csv)} ARCHIVOS ===")

#%%
mapeo_variables = {
    'Playing Time_90s': 'minutos_90s', # Denominador (NO BORRAR)
    
    # --- ATAQUE Y TIROS ---
    'Performance_Gls': 'goles',
    'Standard_Sh': 'tiros_totales',
    'Standard_SoT': 'tiros_al_arco',
    'Standard_Dist': 'distancia_promedio_tiro',
    'Expected_xG': 'xg',
    'Expected_npxG': 'npxg', # xG sin penales
    
    # --- CREACIÓN Y ASISTENCIAS ---
    'Performance_Ast': 'asistencias',
    'Expected_xAG': 'xag', # Asistencias esperadas
    'KP_': 'pases_clave',
    'SCA_SCA': 'acciones_creacion_tiro',
    'GCA_GCA': 'acciones_creacion_gol',
    
    # --- PASES Y DISTRIBUCIÓN ---
    'Total_Att': 'pases_intentados',
    'Total_Cmp': 'pases_completados',
    'Progression_PrgP': 'pases_progresivos',
    '1/3_': 'pases_tercio_ofensivo',
    'PPA_': 'pases_area_rival',
    'Pass Types_Crs': 'centros_totales',
    'CrsPA_': 'centros_area_rival',
    'Short_Cmp': 'pases_cortos_completados',
    'Medium_Cmp': 'pases_medios_completados',
    'Long_Cmp': 'pases_largos_completados',
    'Total_PrgDist': 'distancia_pases_progresivos', # Distancia total en metros hacia el arco
    'Pass Types_TB': 'pases_filtrados',
    'Pass Types_Sw': 'cambios_de_frente',
    
    # --- TRASLADO, REGATES Y POSESIÓN ---
    'Touches_Touches': 'toques_totales',
    'Touches_Att Pen': 'toques_area_rival',
    'Touches_Def Pen': 'toques_area_propia',
    'Take-Ons_Att': 'regates_intentados',
    'Take-Ons_Succ': 'regates_exitosos',
    'Carries_Carries': 'conducciones_totales',
    'Progression_PrgC': 'conducciones_progresivas',
    'Performance_Fld': 'faltas_recibidas',
    'Carries_1/3': 'conducciones_tercio_ofensivo',
    'Carries_CPA': 'conducciones_area_rival',
    'Carries_Dis': 'pelota_perdida', # Cuántas veces le roban la pelota
    'Carries_Mis': 'malos_controles',
    'Receiving_Rec': 'pases_recibidos',
    'Progression_PrgR': 'pases_progresivos_recibidos',
    'Performance_Off': 'fueras_de_juego',
    
    # --- DEFENSA Y RECUPERACIÓN ---
    'Tackles_Tkl': 'quites_totales',
    'Tackles_TklW': 'quites_ganados',
    'Tackles_Def 3rd': 'quites_tercio_defensivo',
    'Tackles_Mid 3rd': 'quites_tercio_medio',
    'Tackles_Att 3rd': 'quites_tercio_ofensivo',
    'Challenges_Tkl': 'dribladores_tackleados',
    'Performance_Fls': 'faltas_cometidas',
    'Int_': 'intercepciones',
    'Performance_Recov': 'recuperaciones',
    'Blocks_Blocks': 'bloqueos_totales',
    'Blocks_Sh': 'tiros_bloqueados',
    'Blocks_Pass': 'pases_bloqueados',
    'Clr_': 'despejes',
    'Err_': 'errores_en_propia_mitad',
    
    
    # --- JUEGO AÉREO ---
    'Aerial Duels_Won': 'duelos_aereos_ganados',
    'Aerial Duels_Lost': 'duelos_aereos_perdidos',
}

# 3. Función procesadora de archivos
def procesar_archivo(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    
    # 3.1 Preprocesamiento de tipos de datos
    text_cols = ['league', 'season', 'team', 'player', 'nation_', 'pos_']
    for col in df.columns:
        if col not in text_cols:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3.2 Filtro de minutos
    if 'Playing Time_Min' not in df.columns:
        return None # Evita quiebres si el archivo está corrupto
    
    df_clean = df[df['Playing Time_Min'] >= 900].copy()
    
    if df_clean.empty:
        return None # Si es una temporada recién empezada y nadie tiene 900 min
        
    # 3.3 Filtro dinámico de columnas (Previene errores en temporadas viejas)
    meta_presentes = [m for m in text_cols if m in df_clean.columns]
    cols_presentes = [c for c in mapeo_variables.keys() if c in df_clean.columns]
    
    df_clean = df_clean[meta_presentes + cols_presentes]
    
    # Solo renombramos las columnas que efectivamente encontramos
    mapeo_real = {k: v for k, v in mapeo_variables.items() if k in cols_presentes}
    df_clean = df_clean.rename(columns=mapeo_real)
    
    # 3.4 Normalización temporal
    vars_directas = ['distancia_promedio_tiro'] 
    vars_directas_presentes = [v for v in vars_directas if v in df_clean.columns]
    
    # Aislamos variables de volumen
    vars_volumen = [col for col in mapeo_real.values() if col not in ['minutos_90s'] + vars_directas_presentes]
    
    df_p90 = df_clean[vars_volumen].div(df_clean['minutos_90s'], axis=0).add_suffix('_p90')
    
    df_clean = pd.concat([df_clean[meta_presentes], df_clean[vars_directas_presentes], df_p90], axis=1)
    
    # 3.5 Estandarización relativa (Z-Score)
    cols_modelo = df_p90.columns.tolist() + vars_directas_presentes
    df_clean[cols_modelo] = df_clean.groupby(['league', 'season'])[cols_modelo].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # 3.6 Indexado e Imputación
    df_model = df_clean.set_index(meta_presentes)
    df_model = df_model.fillna(0)
    
    return df_model
#%%
# 4. Bucle principal de ejecución
lista_procesados = []

for archivo in archivos_csv:
    nombre_base = os.path.basename(archivo)
    print(f"Procesando: {nombre_base}...")
    
    df_procesado = procesar_archivo(archivo)
    
    if df_procesado is not None:
        # Guardamos el CSV individual procesado
        ruta_salida = os.path.join(carpeta_save, f"procesado_{nombre_base}")
        df_procesado.to_csv(ruta_salida)
        
        # Agregamos a la lista maestra
        lista_procesados.append(df_procesado)
    else:
        print(f" -> Saltando {nombre_base} (Sin datos suficientes).")

# 5. Concatenación final de todo el dataset
if lista_procesados:
    df_master = pd.concat(lista_procesados, axis=0)
    
    # Para el PCA necesitamos que todas las columnas coincidan. Si un año no tuvo xG, 
    # pd.concat genera NaNs en esas filas. Imputamos con 0 (media estandarizada).
    df_master = df_master.fillna(0)
    
    print("\n=== PROCESAMIENTO FINALIZADO ===")
    print(f"Total de jugadores analizados (>900 min): {len(df_master)}")
    print(f"Métricas listas para PCA: {len(df_master.columns)}")
    
    # Podés guardar la matriz final completa
    df_master.to_csv(f"{carpeta_save}/matriz_master_pca.csv")
else:
    print("No se procesó ningún archivo.")
# %%
# para ver el dataframe final
df_master