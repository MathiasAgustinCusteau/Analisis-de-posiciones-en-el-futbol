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

3. **Clustering Probabilístico y Evaluación (`Clustering/clustering`):**
   * Comparación entre **K-Means** y **Gaussian Mixture Models (GMM)** mediante Coeficiente de Silueta, Índice de Davies-Bouldin, Criterio de Información Bayesiano (BIC) y estabilidad por semilla (*Adjusted Rand Index*).

4. **Estabilidad del modelo inicial (`estabilidad_gmm.py`):**
   * El primer modelo era un único **GMM con $k = 8$** (semilla 42). Al repetirlo con otras semillas y con submuestras del 80% de los datos, los roles cambiaban: solo *Defensor Central* y *Delantero Finalizador* eran estables (Jaccard > 0,9) y *Generador Élite* se disolvía (Jaccard 0,49). Además, 7 de 10 semillas encontraban una solución con mejor ajuste que la de la semilla 42 y que coincidía con ella solo en un 58% (ARI).

5. **Modelo final: Clustering de Consenso con $k = 7$ (`consenso.py`):**
   * Se ajustan **200 GMM** (covarianza completa), cada uno con una semilla distinta y el 80% de los registros elegidos al azar, y se cuenta en qué fracción de las corridas cada par de jugadores cae en el mismo grupo (**matriz de co-asociación**). Los roles finales salen de agrupar esa matriz con enlace promedio (*evidence accumulation*).
   * **Elección de k** por **PAC** (proporción de pares ambiguos; menor es mejor): k=6: 0,212 · **k=7: 0,113** · k=8: 0,147 · k=9: 0,156.
   * **Reproducibilidad:** el consenso armado con las corridas 1–100 y con las 101–200 por separado coincide en un 94,5% (ARI). Ya no depende de la semilla.
   * Con los grupos de consenso se estima un GMM (peso, media y covarianza de cada rol), que da las **probabilidades de pertenencia** (*soft clustering*, usadas por el buscador de reemplazos) y la **log-densidad** para detectar *outliers*. Coincide con el rol de consenso en el 98,9% de los registros.

---

## Principales Hallazgos

* **7 Roles Tácticos Estables:** *Defensor Central*, *Lateral / Carrilero*, *Volante Recuperador*, *Organizador / Regista*, *Extremo / Creador*, *Segunda Punta / Delantero Móvil* y *Delantero Finalizador (9)*. Los más sólidos son Defensor Central (confianza 0,95), Lateral (0,94) y Delantero Finalizador (0,93); los de bordes más difusos son los ofensivos: Extremo / Creador (0,78) y Segunda Punta (0,71). La *Segunda Punta* queda, métrica por métrica, a mitad de camino entre un 9 y un extremo.
* **Versatilidad Posicional:** Los delanteros puros (`FW`) presentan la menor entropía (`0.466`), concentrándose mayoritariamente en roles de finalización, mientras que los perfiles híbridos defensivos-mediocampistas (`DF,MF` con `0.762`) muestran la mayor dispersión funcional.
* **Predicción (`Prediccion/`):**
  * *Transiciones:* el 69% de los jugadores repite su rol al año siguiente (16% si fuera al azar). Los que cambian de liga lo repiten igual que los que cambian de equipo dentro de su liga (65%), lo que respalda la estandarización por liga. Una regresión logística entrenada solo con temporadas anteriores predice el rol del año siguiente mejor que "repite su rol" en las 5 temporadas evaluadas (71,1% vs 68,1%; mejora significativa en 4 de 5) y anticipa el 26% de los cambios de rol. Incluye un pronóstico para 2025/26.
  * *Edad:* la probabilidad de cambiar de rol baja con la edad (37% a los 19 años, 29% a los 31, a igual rol, confianza y liga). Siguiendo a los mismos 1.209 jugadores durante 4+ años, el rol *Extremo / Creador* pierde 7 puntos y crecen *Segunda Punta*, *Volante* y *Organizador*: los extremos se van retrasando o centralizando.
  * *Clasificador de posición:* con las 55 métricas se adivina la posición nominal (DF/MF/FW) en el 93% de los casos (validación cruzada agrupada por jugador). Los errores se concentran en las posiciones híbridas (23% vs 1,2% en las puras) y en los roles de bordes difusos del consenso (Extremo 21%, Segunda Punta 17%, Central 0,4%): la correlación entre el error y la confianza del rol es −0,91. Lo supervisado y lo no supervisado se validan mutuamente.
* **Robustez al estilo de equipo (`Robustez/`):** un jugador de un equipo que tiene mucho la pelota acumula más pases aunque cumpla la misma función. Se estimó el estilo de cada equipo-temporada (pases por partido respecto de su liga, sumando a todos sus jugadores), se le quitó a cada métrica la parte que explica ese estilo y se repitió el pipeline. El estilo explica bastante de las métricas de pase (48% de *pases recibidos*, 40% de *pases cortos completados*) pero poco del resto (mediana 3%). Con la matriz ajustada, **el 90% de los jugadores mantiene su rol** (ARI 0,81), y Trent, Rodri y Messi siguen igual. Los cambios se concentran en los equipos con muchísima pelota (15% vs 7–10% en el resto) y en el rol *Organizador* (20%): parte de los laterales de equipos de posesión pasan a *Lateral* al quitar el efecto del equipo.
* **Mapa 2D (`mapa_2d.py`, t-SNE):** proyectando los 16 componentes del PCA a un plano, cada rol ocupa su propia zona; Defensor Central, Lateral y Delantero 9 forman zonas compactas, y Extremo y Segunda Punta comparten zona. Los roles se calculan en las 16 dimensiones: el plano es solo para visualizarlos (sus ejes no son la cancha).
* **Validación de Casos Reales:** El algoritmo captura evoluciones tácticas y roles atípicos sin supervisión previa: *Trent Alexander-Arnold* es *Organizador / Regista* en las 8 temporadas pese a figurar como `DF`; *Rodri* pasa de *Volante Recuperador* a *Organizador* justo en 2023/24; y *Lionel Messi 2018/19* es el registro más atípico de todo el dataset (menor log-densidad).


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
│   ├── datos_pca.parquet       # Matriz de componentes principales lista para modelado
│   ├── consenso.py             # MODELO FINAL: consenso de 200 GMM (k=7) -> asignacion_clusters.csv
│   ├── clustering              # Diagnóstico de k (K-Means vs GMM), entropía, perfiles, outliers y gráficos
│   ├── estabilidad_gmm.py      # Estabilidad del modelo inicial (GMM único, k=8): motivó el consenso
│   ├── reemplazos.py           # Buscador de reemplazos (casos Trent y Rodri)
│   ├── figuras_ppt.py          # Figuras de la presentación (Clustering/figuras/), incluye radar por rol
│   ├── mapa_2d.py              # Mapa 2D de los jugadores (t-SNE) por rol; guarda mapa_2d.csv
│   ├── cluster_profiles.py     # Perfiles promedio por rol (lo usa figuras_ppt.py)
│   ├── asignacion_clusters.csv # Rol, probabilidades y confianza de cada registro (modelo final)
│   └── asignacion_clusters_k8_original.csv  # Copia del modelo inicial (k=8), para comparar
├── Prediccion/
│   ├── comun.py                # Carga de datos y estilo compartidos
│   ├── transiciones.py         # Matriz de transición, predicción del rol siguiente y pronóstico 2025/26
│   ├── edad_roles.py           # Edad y cambio de rol
│   ├── clasificador_posicion.py # Clasificador supervisado de la posición nominal
│   └── figuras/                # M_transiciones_roles, N_edad_y_roles, O_clasificador_posicion
├── Robustez/
│   ├── ajuste_estilo_equipo.py # Repite el pipeline quitando el efecto del estilo de equipo
│   └── figuras/                # P_ajuste_estilo_equipo
├── tests/                      # Tests de cluster_profiles.py
├── requirements.txt            # Hacer pip install -r requirements.txt para facilitar la preparacion de tu .venv
└── README.md
```

## Orden de ejecución

Desde la raíz del repo (los datos procesados ya están generados, no hace falta volver a descargarlos):

```bash
python "preparacion de datos/pca_global.py"   # solo si cambian los datos procesados
python "Clustering/consenso.py"               # modelo final (~5 min)
python "Clustering/clustering"                # diagnóstico, entropía, perfiles y outliers
python "Clustering/reemplazos.py"
python "Clustering/figuras_ppt.py"
python "Clustering/mapa_2d.py"                # ~1 min la primera vez (t-SNE)
python "Clustering/estabilidad_gmm.py"        # opcional (~10 min): evalúa el modelo inicial k=8
python "Prediccion/transiciones.py"
python "Prediccion/edad_roles.py"             # ~2 min (bootstrap)
python "Prediccion/clasificador_posicion.py"
python "Robustez/ajuste_estilo_equipo.py"     # ~4 min
```

## Perfiles de los siete roles

El script guarda `Clustering/perfiles_clusters.csv` con el promedio de cada métrica por clúster, `N_registros` (filas jugador-equipo-temporada) y `N_jugadores_unicos`. También genera `perfiles_clusters_resumen.csv` con las métricas promedio más altas y más bajas, además de `figuras/I_tabla_perfiles_clusters.png` y `figuras/J_heatmap_perfiles_clusters.png`. Las métricas son Z-scores: valores positivos indican un promedio por encima de su liga y temporada; negativos, por debajo.

