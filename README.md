# Análisis de Roles Tácticos y Versatilidad en el Fútbol Europeo (2017–2025)

Proyecto final de análisis y ciencia de datos enfocado en descubrir arquetipos reales de rendimiento en las **5 grandes ligas europeas** (Premier League, La Liga, Serie A, Bundesliga y Ligue 1) entre las temporadas **2017/18 y 2024/25**, utilizando aprendizaje no supervisado.

## Objetivo del Proyecto

Las etiquetas posicionales tradicionales (`DF`, `MF`, `FW`) suelen ocultar la verdadera función táctica de un futbolista dentro del campo. Este proyecto invierte el enfoque clásico:

1. **Agrupamiento puramente por rendimiento:** Se segmenta a los jugadores utilizando exclusivamente sus métricas estadísticas en cancha, sin proporcionarles información sobre su posición nominal.
2. **Cruce con posiciones reales y medición de versatilidad:** Se analiza cómo se distribuye cada posición nominal a través de los clústeres descubiertos, cuantificando la versatilidad funcional mediante la **Entropía de Shannon**.
3. **Detección de jugadores híbridos y *outliers*:** Identificación de futbolistas que rompen el molde de su posición nominal (ej. laterales que juegan como mediocampistas organizadores o creadores de élite con métricas extremas).


## Metodología y Pipeline de Datos

1. **Preprocesamiento y Estandarización Relativa (`procesar_datos.py`):**
   * Filtrado de jugadores de campo con al menos **900 minutos** disputados por temporada (excluyendo arqueros `GK` para evitar sesgos de escala).
   * Normalización temporal (*per 90 minutes*) sobre más de 50 métricas ofensivas, creativas, de pase, traslado, defensa y juego aéreo.
   * Estandarización relativa (**Z-Score intra-liga e intra-temporada**) para hacer comparables rendimientos entre distintas ligas y años, consolidando **11.889 registros jugador-temporada**.

2. **Reducción de Dimensionalidad (`pca_global.py`):**
   * Aplicación de **PCA Global**, reteniendo **16 componentes principales** que explican el **90% de la varianza** total del dataset.

3. **Clustering Probabilístico y Evaluación (`exploracion_inicial.py` y `clustering.py`):**
   * Comparación entre **K-Means** y **Gaussian Mixture Models (GMM)** mediante Coeficiente de Silueta, Índice de Davies-Bouldin, Criterio de Información Bayesiano (BIC) y estabilidad por semilla (*Adjusted Rand Index*).
   * Selección de **GMM con covarianza completa ($k = 8$)**, permitiendo tanto la asignación a 8 arquetipos tácticos definidos como el cálculo de probabilidades de pertenencia (*soft clustering*) y log-densidad para detección de *outliers*.

---

## Principales Hallazgos

* **8 Arquetipos Tácticos Descubiertos:** El modelo separa de forma natural perfiles como *Defensor Central Clásico*, *Lateral/Carrilero*, *Volante Recuperador*, *Organizador/Regista*, *Mediapunta de Transición*, *Generador Élite*, *Delantero Finalizador (9)* y *Pivot Físico*.
* **Versatilidad Posicional:** Los delanteros puros (`FW`) presentan la menor entropía (`0.484`), concentrándose mayoritariamente en roles de finalización, mientras que los perfiles híbridos defensivos-mediocampistas (`DF,MF` con `0.774`) muestran la mayor dispersión funcional.
* **Validación de Casos Reales:** El algoritmo captura evoluciones tácticas y roles atípicos sin supervisión previa (ej. *Trent Alexander-Arnold* clasificado consistentemente como mediocampista organizador pese a figurar como `DF`, el salto de rol de *Rodri* en 2023/24, y *Lionel Messi 2018/19* como extremo estadístico dentro del clúster de creadores de élite). 


---

## Estructura del Repositorio

```text
├── datos_kaggle/
│   ├── ligas_temporadas/       # Archivos CSV crudos por liga y temporada
│   └── procesados/      # Datasets normalizados y matriz maestra lista para PCA
├── preparacion de datos/
│   ├── data_kaggle.py          # Descarga de datos desde Kaggle
│   ├── procesar_datos.py       # Limpieza, filtro de minutos (>900 min), exclusión de GK y Z-Score
│   └── pca_global.py           # Reducción de dimensionalidad (PCA) y exportación a Parquet
├── Clustering/
│   ├── clustering.py           # Evaluación comparativa de modelos y métricas. modelo GMM final (k=8), cálculo de entropía, perfiles tácticos y gráficos
│   └── datos_pca.parquet       # Matriz de componentes principales lista para modelado
├── requirements.txt            # Hacer pip install -r requirements.txt para facilitar la preparacion de tu .venv
└── README.md

