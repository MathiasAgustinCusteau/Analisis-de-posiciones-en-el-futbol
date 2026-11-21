#%%
# =====================================================================
#  FIGURAS PARA LA PRESENTACIÓN
#  Requiere haber corrido Clustering/consenso.py (asignacion_clusters.csv).
#  Las figuras se guardan en Clustering/figuras/.
#
#  Criterios (clase 4 de visualización):
#   - Proporciones / magnitudes -> escala secuencial de un solo color.
#   - Z-scores (centrados en 0) -> eje con el 0 marcado, colores categóricos
#     solo para identificar grupos (paleta apta para daltónicos, validada).
#   - Texto grande para que se lea proyectado.
# =====================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from cluster_profiles import build_cluster_profiles, build_cluster_summary

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
carpeta_fig = RAIZ_PROYECTO / 'Clustering' / 'figuras'
carpeta_fig.mkdir(exist_ok=True)

META = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']
df_z = pd.read_csv(RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv')
asig = pd.read_csv(RAIZ_PROYECTO / 'Clustering' / 'asignacion_clusters.csv')
VARIABLES = [c for c in df_z.columns if c not in META]
Z = df_z[VARIABLES].to_numpy()
perfiles_clusters = build_cluster_profiles(df_z, asig, VARIABLES)
K = asig['Cluster'].nunique()
if perfiles_clusters['Cluster'].tolist() != list(range(K)):
    raise ValueError(
        f"Se esperaban los clústeres 0–{K - 1}; se encontraron {perfiles_clusters['Cluster'].tolist()}"
    )
NUM_TEXTO = {6: 'seis', 7: 'siete', 8: 'ocho', 9: 'nueve'}.get(K, str(K))


# --- Estilo común ---
TINTA = '#0b0b0b'
TINTA_2 = '#52514e'
GRILLA = '#e4e3df'
SUPERFICIE = '#ffffff'
AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'   # validados juntos (daltonismo)
GRIS = '#a3a29c'
SECUENCIAL = LinearSegmentedColormap.from_list(
    'azules', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 17, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA,
    'axes.edgecolor': GRILLA, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# Nombres de los roles (salen de consenso.py)
NOMBRE_ROL = {c: r.split(': ', 1)[1] for c, r in
              asig.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index().items()}
ID_ROL = {nombre: c for c, nombre in NOMBRE_ROL.items()}


def corto(nombre):
    """Nombre en dos renglones para los ejes."""
    if ' / ' in nombre:
        a, b = nombre.split(' / ', 1)
        return f'{a}\n/ {b}'
    return nombre.replace(' ', '\n', 1)


NOMBRE_CORTO = {c: corto(nombre) for c, nombre in NOMBRE_ROL.items()}
# consenso.py ya numera los roles "de atrás hacia adelante": la tabla queda con una diagonal legible
ORDEN_CLUSTERS = list(range(K))
ORDEN_POS = ['DF', 'DF,MF', 'DF,FW', 'MF,DF', 'MF', 'MF,FW', 'FW,MF', 'FW,DF', 'FW']
ETIQ_POS = {'DF': 'DF', 'DF,MF': 'DF / MF', 'DF,FW': 'DF / FW', 'MF,DF': 'MF / DF', 'MF': 'MF',
            'MF,FW': 'MF / FW', 'FW,MF': 'FW / MF', 'FW,DF': 'FW / DF', 'FW': 'FW'}


def etiqueta(v):
    return v.replace('_p90', '').replace('_', ' ')


#%%
# ---------------------------------------------------------------------
# FIGURA A: ¿En qué rol cae cada posición nominal? (reemplaza al heatmap viejo)
# ---------------------------------------------------------------------
tabla = pd.crosstab(asig['pos_'], asig['Cluster'], normalize='index') * 100
tabla = tabla.loc[ORDEN_POS, ORDEN_CLUSTERS]
n_pos = asig['pos_'].value_counts().loc[ORDEN_POS]

fig, ax = plt.subplots(figsize=(13, 6.8))
ax.imshow(np.where(tabla.values < 1, 0, tabla.values), cmap=SECUENCIAL, vmin=0, vmax=70, aspect='auto')
for i in range(tabla.shape[0]):
    for j in range(tabla.shape[1]):
        v = tabla.values[i, j]
        if v >= 1:
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=12,
                    color='white' if v >= 35 else TINTA, fontweight='bold' if v >= 35 else 'normal')
# grilla blanca entre celdas
ax.set_xticks(np.arange(-.5, tabla.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, tabla.shape[0]), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='both', length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(range(len(ORDEN_CLUSTERS)), [NOMBRE_CORTO[c] for c in ORDEN_CLUSTERS], fontsize=12)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ORDEN_POS)), [f'{ETIQ_POS[p]}  (n={n_pos[p]:,})'.replace(',', '.') for p in ORDEN_POS],
              fontsize=12)
ax.set_title('¿En qué rol táctico termina cada posición nominal?\n', pad=4)
fig.text(0.125, 0.04, 'Cada fila suma 100%: de todos los jugadores con esa posición, qué % cae en cada rol. '
         'Celdas vacías: menos de 1%.\nRoles ordenados de atrás hacia adelante. '
         '5 grandes ligas, 2017/18–2024/25, jugadores con ≥900 min.',
         fontsize=10.5, color=TINTA_2)
fig.savefig(carpeta_fig / 'A_posiciones_vs_roles.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# FIGURA B: la "Segunda Punta" queda entre el 9 y el extremo
# ---------------------------------------------------------------------
vars_b = ['fueras_de_juego_p90', 'xg_p90', 'toques_area_rival_p90', 'tiros_totales_p90',
          'duelos_aereos_ganados_p90', 'malos_controles_p90',
          'regates_intentados_p90', 'conducciones_progresivas_p90', 'pases_clave_p90',
          'acciones_creacion_tiro_p90', 'toques_totales_p90', 'pases_completados_p90', 'recuperaciones_p90']
perfil = pd.DataFrame(Z, columns=VARIABLES).groupby(asig['Cluster'].values).mean()
grupos = [(ID_ROL[nombre], nombre, color, marcador) for nombre, color, marcador in [
    ('Delantero Finalizador (9)', AZUL, 'o'),
    ('Segunda Punta / Delantero Móvil', NARANJA, 'D'),
    ('Extremo / Creador', AQUA, 's')]]

fig, ax = plt.subplots(figsize=(11, 7.5))
y = np.arange(len(vars_b))[::-1]
for v_i, v in zip(y, vars_b):
    vals = [perfil.loc[c, v] for c, *_ in grupos]
    ax.plot([min(vals), max(vals)], [v_i, v_i], color=GRILLA, linewidth=3, zorder=1)
for c, nombre, color, marcador in grupos:
    ax.scatter(perfil.loc[c, vars_b], y, s=110, color=color, marker=marcador, label=nombre,
               edgecolor='white', linewidth=1.5, zorder=3)
ax.axvline(0, color=TINTA_2, linewidth=1, zorder=0)
ax.text(0.03, -0.75, 'promedio de la liga', fontsize=10, color=TINTA_2)
ax.set_ylim(-1, len(vars_b) - 0.5)
ax.set_yticks(y, [etiqueta(v) for v in vars_b])
ax.set_xlabel('Diferencia con el promedio de su liga y temporada (desvíos estándar, por 90 min)')
ax.grid(axis='x', color=GRILLA, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(loc='lower right', frameon=False, fontsize=12)
ax.set_title('La "Segunda Punta" juega a mitad de camino entre un 9 y un extremo')
fig.savefig(carpeta_fig / 'B_segunda_punta_vs_9_vs_extremo.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# FIGURA C: Reemplazos de Trent (2024/25) y dónde quedó Frimpong
# ---------------------------------------------------------------------
def ranking_reemplazos(id_jugador, temporada, temporada_busqueda, rol=None, prob_min=0.5):
    i = asig.index[(asig['id_jugador'] == id_jugador) & (asig['season'] == temporada)][0]
    rol = int(asig.loc[i, 'Cluster']) if rol is None else rol   # por defecto, el rol del jugador
    pos_principal = asig.loc[i, 'pos_'].split(',')[0]
    cand = asig[(asig['season'] == temporada_busqueda) & (asig['id_jugador'] != id_jugador)
                & (asig['team'] != asig.loc[i, 'team'])].copy()
    cand['Distancia'] = np.linalg.norm(Z[cand.index] - Z[i], axis=1)
    cand['Puesto'] = cand['Distancia'].rank(method='first').astype(int)
    cand['Adaptable'] = ~cand['pos_'].str.split(',').apply(lambda p: pos_principal in p)
    cand['En_rol'] = cand[f'Prob_C{rol}'] >= prob_min
    return cand.sort_values('Distancia')


def figura_reemplazos(id_jugador, temporada, temporada_busqueda, titulo, archivo, rol=None, extra=None, n=10):
    cand = ranking_reemplazos(id_jugador, temporada, temporada_busqueda, rol)
    top = cand[cand['En_rol']].head(n)
    filas = top.copy()
    if extra is not None:
        filas = pd.concat([filas, cand[cand['id_jugador'] == extra]])
    filas = filas.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(filas) + 1.8))
    yy = np.arange(len(filas))
    for k, (_, r) in enumerate(filas.iterrows()):
        es_extra = extra is not None and r['id_jugador'] == extra
        color = GRIS if es_extra else (NARANJA if r['Adaptable'] else AZUL)
        ax.plot([0, r['Distancia']], [k, k], color=color, linewidth=2, alpha=0.5)
        ax.scatter(r['Distancia'], k, s=90, color=color, zorder=3, edgecolor='white', linewidth=1.5)
        ax.text(r['Distancia'] + 0.12, k, f"{r['Distancia']:.1f}", va='center', fontsize=11, color=TINTA_2)
    nombres = [f"{r['player']}  ·  {r['team']}  ({r['pos_'].replace(',', '/')})"
               + (f"   — puesto {r['Puesto']} de {len(cand)}" if extra is not None and r['id_jugador'] == extra else '')
               for _, r in filas.iterrows()]
    ax.set_yticks(yy, nombres, fontsize=12)
    ax.tick_params(length=0)
    ax.set_xlim(0, filas['Distancia'].max() * 1.12)
    ax.set_xlabel('Distancia al perfil del jugador (menor = más parecido)')
    ax.grid(axis='x', color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    # leyenda manual
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', color=AZUL, markersize=9, label='Misma posición nominal')]
    if top['Adaptable'].any():
        handles.append(Line2D([], [], marker='o', linestyle='', color=NARANJA, markersize=9,
                              label='Otra posición (adaptable)'))
    if extra is not None:
        handles.append(Line2D([], [], marker='o', linestyle='', color=GRIS, markersize=9, label='Fichaje real'))
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=11)
    ax.set_title(titulo)
    fig.savefig(carpeta_fig / archivo)
    plt.show()


figura_reemplazos('Trent Alexander-Arnold (1998)', 2425, 2425,
                  titulo='Se va Trent: los 10 jugadores de 2024/25 más parecidos a él',
                  archivo='C_reemplazos_trent.png', extra='Jeremie Frimpong (2000)')
figura_reemplazos('Rodri (1996)', 2324, 2425,
                  titulo='Se lesiona Rodri: los 10 jugadores de 2024/25 más parecidos a su 2023/24',
                  archivo='D_reemplazos_rodri.png')
# %%

#%%
# ---------------------------------------------------------------------
# FIGURA E: Versatilidad por posición (entropía de Shannon normalizada)
# Un solo color: la longitud de la barra ya codifica el valor; pintar cada
# barra de un color distinto (viridis) no agrega información y sugiere
# categorías que no existen.
# ---------------------------------------------------------------------
from scipy.stats import entropy

tabla_e = pd.crosstab(asig['pos_'], asig['Cluster'], normalize='index')
ent = tabla_e.apply(lambda r: entropy(r, base=len(r)), axis=1).sort_values()
n_e = asig['pos_'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5.5))
yy = np.arange(len(ent))
ax.barh(yy, ent.values, color=AZUL, height=0.62)
for k, (p, v) in enumerate(ent.items()):
    ax.text(v + 0.01, k, f'{v:.2f}', va='center', fontsize=12, color=TINTA)
ax.set_yticks(yy, [f'{ETIQ_POS[p]}  (n={n_e[p]:,})'.replace(',', '.') for p in ent.index], fontsize=12)
ax.set_xlim(0, 1)
ax.set_xlabel(f'Entropía normalizada\n0 = todos caen en un mismo rol  ·  1 = repartidos en partes iguales en los {K}')
ax.grid(axis='x', color=GRILLA, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.set_title(f'¿Qué tan repartida está cada posición entre los {K} roles?')
fig.savefig(carpeta_fig / 'E_versatilidad_entropia.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# FIGURA F: Elección de k (sin doble eje: un panel por criterio)
# ---------------------------------------------------------------------
ruta_diag = RAIZ_PROYECTO / 'Clustering' / 'diagnostico_k.csv'
if ruta_diag.exists():
    diag = pd.read_csv(ruta_diag)
    paneles = [
        ('GMM_BIC', 'BIC del GMM\n(menor = mejor ajuste)', AZUL),
        ('GMM_Sil', 'Silueta del GMM\n(mayor = grupos más separados)', AZUL),
        ('KM_Estab(ARI)', 'Estabilidad de K-Means entre semillas\n(ARI, 1 = misma partición)', NARANJA),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (col, titulo, color) in zip(axes, paneles):
        ax.plot(diag['k'], diag[col], marker='o', color=color, linewidth=2, markersize=8)
        sel = diag[diag['k'] == K]
        ax.scatter(sel['k'], sel[col], s=160, facecolor='none', edgecolor=TINTA, linewidth=2, zorder=4)
        ax.axvline(K, color=GRIS, linestyle='--', linewidth=1, zorder=0)
        ax.set_title(titulo, fontsize=13)
        ax.grid(color=GRILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(diag['k'])
        if col == 'KM_Estab(ARI)':
            ax.set_ylim(0.9, 1.005)   # no exagerar diferencias mínimas
        ax.set_xlabel('k (cantidad de clústeres)')
        ax.tick_params(length=0)
    fig.suptitle(f'¿Cuántos roles? Tres criterios para elegir k (k = {K} marcado; ver también L_consenso.png)', x=0.01, ha='left',
                 fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(carpeta_fig / 'F_seleccion_k.png')
    plt.show()
else:
    print("Falta diagnostico_k.csv: corré Clustering/clustering primero.")

#%%
# ---------------------------------------------------------------------
# FIGURA G: Varianza explicada por el PCA
# ---------------------------------------------------------------------
from sklearn.decomposition import PCA

var_acum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
n90 = int(np.argmax(var_acum >= 0.90) + 1)
fig, ax = plt.subplots(figsize=(10, 5))
k_eje = np.arange(1, len(var_acum) + 1)
ax.plot(k_eje, var_acum * 100, color=AZUL, linewidth=2, marker='o', markersize=4)
ax.axhline(90, color=GRIS, linestyle='--', linewidth=1)
ax.scatter([n90], [var_acum[n90 - 1] * 100], s=160, facecolor='none', edgecolor=TINTA, linewidth=2, zorder=4)
ax.annotate(f'{n90} componentes → 90% de la varianza', (n90, var_acum[n90 - 1] * 100),
            xytext=(n90 + 4, 72), fontsize=12, arrowprops=dict(arrowstyle='-', color=TINTA_2))
ax.set_xlabel('Cantidad de componentes principales')
ax.set_ylabel('Varianza explicada acumulada (%)')
ax.set_ylim(0, 102)
ax.grid(color=GRILLA, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.set_title(f'De {Z.shape[1]} métricas a {n90} componentes')
fig.savefig(carpeta_fig / 'G_varianza_pca.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# FIGURA H: Trent vs Pedro Porro (2024/25), métrica por métrica
# Se muestran las métricas en las que Trent más se destaca (|z| más grande).
# ---------------------------------------------------------------------
def fila_z(id_jugador, temporada):
    i = asig.index[(asig['id_jugador'] == id_jugador) & (asig['season'] == temporada)][0]
    return pd.Series(Z[i], index=VARIABLES)


trent = fila_z('Trent Alexander-Arnold (1998)', 2425)
porro = fila_z('Pedro Porro (1999)', 2425)
vars_h = trent.abs().sort_values(ascending=False).head(16).index
vars_h = trent[vars_h].sort_values().index   # ordenadas por el valor de Trent

fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(vars_h))
for k, v in enumerate(vars_h):
    ax.plot([trent[v], porro[v]], [k + 0.13, k - 0.13], color=GRILLA, linewidth=3, zorder=1)
ax.scatter(trent[vars_h], y + 0.13, s=120, color=AZUL, marker='o', label='Trent Alexander-Arnold (Liverpool)',
           edgecolor='white', linewidth=1.5, zorder=3)
ax.scatter(porro[vars_h], y - 0.13, s=120, color=NARANJA, marker='D', label='Pedro Porro (Tottenham)',
           edgecolor='white', linewidth=1.5, zorder=3)
ax.axvline(0, color=TINTA_2, linewidth=1, zorder=0)
ax.text(0.05, -0.9, 'promedio de la liga', fontsize=10, color=TINTA_2)
ax.set_ylim(-1.2, len(vars_h) - 0.4)
ax.set_yticks(y, [etiqueta(v) for v in vars_h])
ax.set_xlabel('Diferencia con el promedio de su liga y temporada (desvíos estándar, por 90 min)')
ax.grid(axis='x', color=GRILLA, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(loc='lower right', frameon=False, fontsize=12)
ax.set_title('Trent vs. Porro (2024/25): mismo perfil, distinta intensidad')
fig.savefig(carpeta_fig / 'H_trent_vs_porro.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# PERFILES DE LOS ROLES: CSV, tabla resumen y mapa de calor
# ---------------------------------------------------------------------
perfil_resumen = build_cluster_summary(perfiles_clusters, VARIABLES, top_n=3, bottom_n=2)
ruta_perfiles = RAIZ_PROYECTO / 'Clustering' / 'perfiles_clusters.csv'
ruta_resumen = RAIZ_PROYECTO / 'Clustering' / 'perfiles_clusters_resumen.csv'
perfiles_clusters.to_csv(ruta_perfiles, index=False)
perfil_resumen.to_csv(ruta_resumen, index=False)

# Tabla legible: tamaño de cada grupo y métricas promedio más altas/bajas.
def saltos_de_linea_metricas(texto):
    metricas = []
    for item in texto.split(';'):
        nombre, valor = item.strip().rsplit(' (', 1)
        metricas.append(f"{etiqueta(nombre)} ({valor}")
    return '\n'.join(metricas)

filas_tabla = []
for _, fila in perfil_resumen.iterrows():
    rol_corto = str(fila['Rol']).split(': ', 1)[-1]
    filas_tabla.append([
        f"C{int(fila['Cluster'])} · {rol_corto}",
        f"{int(fila['N_registros']):,}".replace(',', '.'),
        f"{int(fila['N_jugadores_unicos']):,}".replace(',', '.'),
        saltos_de_linea_metricas(fila['metricas_mas_altas']),
        saltos_de_linea_metricas(fila['metricas_mas_bajas']),
    ])

fig, ax = plt.subplots(figsize=(18, 9))
ax.axis('off')
tabla = ax.table(
    cellText=filas_tabla,
    colLabels=['Clúster / rol', 'Registros\njugador-equipo-temp.', 'Jugadores\núnicos',
               'Métricas promedio más altas (Z)', 'Métricas promedio más bajas (Z)'],
    cellLoc='left',
    colLoc='center',
    colWidths=[0.19, 0.12, 0.10, 0.32, 0.27],
    bbox=[0, 0.08, 1, 0.82],
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 1.7)
for (fila, columna), celda in tabla.get_celld().items():
    celda.set_edgecolor(GRILLA)
    if fila == 0:
        celda.set_facecolor('#184f95')
        celda.set_text_props(color='white', weight='bold', ha='center', va='center')
    else:
        celda.set_facecolor('#f4f7fb' if fila % 2 else 'white')
        celda.set_text_props(color=TINTA, va='center')
        if columna in (1, 2):
            celda.set_text_props(ha='center')
ax.set_title(f'{NUM_TEXTO.capitalize()} perfiles estadísticos de rendimiento', loc='left', pad=18,
             fontsize=20, fontweight='bold', color=TINTA)
fig.text(
    0.01, 0.025,
    'Z = promedio del clúster respecto de su liga y temporada; valores positivos indican métricas sobre ese promedio. '
    'Los registros cuentan jugador-equipo-temporada; jugadores únicos deduplican por id_jugador.',
    fontsize=10, color=TINTA_2,
)
fig.savefig(carpeta_fig / 'I_tabla_perfiles_clusters.png')
plt.show()

# Mapa de calor con todas las métricas para comparar los roles.
matriz_heatmap = perfiles_clusters.set_index('Cluster')[VARIABLES].T
nombres_eje_x = []
for _, fila in perfiles_clusters.iterrows():
    rol_corto = str(fila['Rol']).split(': ', 1)[-1].replace(' / ', ' /\n')
    nombres_eje_x.append(f"C{int(fila['Cluster'])}\n{rol_corto}")
matriz_heatmap.columns = nombres_eje_x
matriz_heatmap.index = [etiqueta(nombre) for nombre in matriz_heatmap.index]

fig, ax = plt.subplots(figsize=(13, 17))
sns.heatmap(
    matriz_heatmap,
    ax=ax,
    cmap='vlag',
    center=0,
    vmin=-2.5,
    vmax=2.5,
    linewidths=0.25,
    linecolor='white',
    cbar_kws={'label': 'Promedio Z por clúster (escala ±2,5)'},
    yticklabels=True,
    xticklabels=True,
)
ax.set_title(f'Comparación de métricas en los {NUM_TEXTO} roles', loc='left',
             fontsize=19, fontweight='bold', color=TINTA, pad=16)
ax.set_xlabel('Clúster y rol interpretado')
ax.set_ylabel('Métrica (por 90 minutos, salvo distancia de tiro)')
ax.tick_params(axis='x', labelrotation=0, labelsize=9)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout()
fig.savefig(carpeta_fig / 'J_heatmap_perfiles_clusters.png')
plt.show()

print(f"\nPerfiles completos guardados en: {ruta_perfiles}")
print(f"Resumen de métricas guardado en: {ruta_resumen}")
print(f"Tabla y mapa de calor guardados en: {carpeta_fig}")
# %%

#%%
# ---------------------------------------------------------------------
# FIGURA Q: Radar de cada rol
# Percentil promedio del rol en 11 métricas elegidas para recorrer la cancha
# (defensa -> construcción -> creación -> ataque). Percentil 50 = jugador típico.
# Se usan percentiles y no z-scores porque el radar necesita una escala 0-100.
# ---------------------------------------------------------------------
METRICAS_RADAR = {
    'despejes_p90': 'Despejes', 'duelos_aereos_ganados_p90': 'Duelos\naéreos', 'quites_totales_p90': 'Quites',
    'intercepciones_p90': 'Intercep-\nciones', 'pases_completados_p90': 'Pases', 'pases_progresivos_p90': 'Pases\nprogresivos',
    'centros_totales_p90': 'Centros', 'pases_clave_p90': 'Pases\nclave', 'regates_intentados_p90': 'Regates',
    'toques_area_rival_p90': 'Toques en\nárea rival', 'npxg_p90': 'xG sin\npenales',
}
percentiles = df_z[list(METRICAS_RADAR)].rank(pct=True) * 100
perfil_radar = percentiles.groupby(asig['Cluster'].values).mean()
angulos = np.linspace(0, 2 * np.pi, len(METRICAS_RADAR), endpoint=False)
cerrar = lambda v: np.append(v, v[0])

n_col = 4
n_fil = int(np.ceil((K + 1) / n_col))
fig, axes = plt.subplots(n_fil, n_col, figsize=(17, 4.6 * n_fil), subplot_kw={'projection': 'polar'})
axes = axes.ravel()
for ax, c in zip(axes, range(K)):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(cerrar(angulos), [50] * (len(angulos) + 1), color=GRIS, linewidth=1.2, linestyle='--', zorder=2)
    ax.fill(cerrar(angulos), cerrar(perfil_radar.loc[c].to_numpy()), color=AZUL, alpha=0.25, zorder=3)
    ax.plot(cerrar(angulos), cerrar(perfil_radar.loc[c].to_numpy()), color=AZUL, linewidth=2.2, zorder=4)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75], ['', '', ''])
    ax.set_xticks(angulos, list(METRICAS_RADAR.values()), fontsize=9, color=TINTA_2)
    ax.tick_params(axis='x', pad=2)
    ax.grid(color=GRILLA, linewidth=0.8)
    ax.spines['polar'].set_color(GRILLA)
    n_rol = (asig['Cluster'] == c).sum()
    ax.set_title(f"{NOMBRE_ROL[c]}\n(n={n_rol:,})".replace(',', '.'), fontsize=13, fontweight='bold', pad=18)
for ax in axes[K:]:
    ax.axis('off')
leyenda = axes[K] if K < len(axes) else axes[-1]
leyenda.text(0.5, 0.62, 'Cómo leerlo', ha='center', fontsize=13, fontweight='bold', transform=leyenda.transAxes)
leyenda.text(0.5, 0.30,
             'Cada eje es una métrica por 90 min.\nLa forma azul es el percentil promedio\ndel rol '
             '(0 = el que menos hace,\n100 = el que más).\nLa línea punteada es el percentil 50:\nun jugador típico.',
             ha='center', va='center', fontsize=11, color=TINTA_2, transform=leyenda.transAxes)
fig.suptitle(f'La forma de cada uno de los {K} roles', x=0.01, ha='left', fontsize=19, fontweight='bold')
fig.tight_layout(h_pad=4)
fig.savefig(carpeta_fig / 'Q_radar_roles.png')
plt.show()
# %%
