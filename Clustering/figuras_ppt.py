#%%
# =====================================================================
#  FIGURAS PARA LA PRESENTACIÓN
#  Requiere haber corrido Clustering/clustering (asignacion_clusters.csv).
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
from matplotlib.colors import LinearSegmentedColormap

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
carpeta_fig = RAIZ_PROYECTO / 'Clustering' / 'figuras'
carpeta_fig.mkdir(exist_ok=True)

META = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']
df_z = pd.read_csv(RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv')
asig = pd.read_csv(RAIZ_PROYECTO / 'Clustering' / 'asignacion_clusters.csv')
VARIABLES = [c for c in df_z.columns if c not in META]
Z = df_z[VARIABLES].to_numpy()

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

NOMBRE_CORTO = {
    0: 'Generador\nÉlite', 1: 'Organizador\n/ Regista', 2: 'Volante\nRecuperador', 3: 'Defensor\nCentral',
    4: 'Lateral /\nCarrilero', 5: 'Mediapunta\n/ Extremo', 6: 'Delantero\nFinalizador', 7: 'Pivot Físico\n/ Choque',
}
# Orden "de atrás hacia adelante" para que la tabla tenga una diagonal legible
ORDEN_CLUSTERS = [3, 4, 2, 1, 7, 5, 0, 6]
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
# FIGURA B: el "Pivot Físico" (C7) queda entre el 9 (C6) y el recuperador (C2)
# ---------------------------------------------------------------------
vars_b = ['xg_p90', 'tiros_totales_p90', 'toques_area_rival_p90', 'duelos_aereos_ganados_p90',
          'duelos_aereos_perdidos_p90', 'malos_controles_p90', 'faltas_cometidas_p90',
          'quites_totales_p90', 'intercepciones_p90', 'recuperaciones_p90',
          'pases_completados_p90', 'pases_progresivos_p90']
perfil = pd.DataFrame(Z, columns=VARIABLES).groupby(asig['Cluster'].values).mean()
grupos = [(6, 'Delantero Finalizador (9)', AZUL, 'o'),
          (7, 'Pivot Físico / Choque', NARANJA, 'D'),
          (2, 'Volante Recuperador', AQUA, 's')]

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
ax.set_title('El "Pivot Físico" juega a mitad de camino entre un 9 y un volante')
fig.savefig(carpeta_fig / 'B_pivot_vs_9_vs_volante.png')
plt.show()

#%%
# ---------------------------------------------------------------------
# FIGURA C: Reemplazos de Trent (2024/25) y dónde quedó Frimpong
# ---------------------------------------------------------------------
def ranking_reemplazos(id_jugador, temporada, temporada_busqueda, rol, prob_min=0.5):
    i = asig.index[(asig['id_jugador'] == id_jugador) & (asig['season'] == temporada)][0]
    pos_principal = asig.loc[i, 'pos_'].split(',')[0]
    cand = asig[(asig['season'] == temporada_busqueda) & (asig['id_jugador'] != id_jugador)
                & (asig['team'] != asig.loc[i, 'team'])].copy()
    cand['Distancia'] = np.linalg.norm(Z[cand.index] - Z[i], axis=1)
    cand['Puesto'] = cand['Distancia'].rank(method='first').astype(int)
    cand['Adaptable'] = ~cand['pos_'].str.split(',').apply(lambda p: pos_principal in p)
    cand['En_rol'] = cand[f'Prob_C{rol}'] >= prob_min
    return cand.sort_values('Distancia')


def figura_reemplazos(id_jugador, temporada, temporada_busqueda, rol, titulo, archivo, extra=None, n=10):
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


figura_reemplazos('Trent Alexander-Arnold (1998)', 2425, 2425, rol=1,
                  titulo='Se va Trent: los 10 jugadores de 2024/25 más parecidos a él',
                  archivo='C_reemplazos_trent.png', extra='Jeremie Frimpong (2000)')
figura_reemplazos('Rodri (1996)', 2324, 2425, rol=1,
                  titulo='Se lesiona Rodri: los 10 jugadores de 2024/25 más parecidos a su 2023/24',
                  archivo='D_reemplazos_rodri.png')
# %%
