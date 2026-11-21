#%%
# =====================================================================
#  MAPA 2D DE TODOS LOS JUGADORES (t-SNE)
#
#  t-SNE "aplana" los 16 componentes del PCA a 2 dimensiones intentando que
#  los jugadores parecidos queden cerca. Sirve para VER los roles, no para
#  medirlos: las distancias entre grupos lejanos y los ejes no tienen
#  significado, solo importa quién queda cerca de quién.
#
#  Figura R: un panel por rol (el rol en azul sobre todos los jugadores en
#  gris) + un panel con la trayectoria de Trent, Rodri y Messi.
#  Se usa un panel por rol porque 7 colores en un mismo gráfico de puntos no
#  se distinguen bien (en especial para personas daltónicas).
#
#  Requiere Clustering/consenso.py. El t-SNE tarda ~1 min y se guarda en
#  mapa_2d.csv; si ya existe y coincide con los datos, se reutiliza.
# =====================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
carpeta = RAIZ_PROYECTO / 'Clustering'
carpeta_fig = carpeta / 'figuras'
carpeta_fig.mkdir(exist_ok=True)
ruta_mapa = carpeta / 'mapa_2d.csv'
RECALCULAR = False      # True para forzar un t-SNE nuevo

AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'    # paleta validada (daltonismo)
GRIS_PUNTO, TINTA, TINTA_2, GRILLA = '#c9c8c2', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

X = pd.read_parquet(carpeta / 'datos_pca.parquet').to_numpy()
asig = pd.read_csv(carpeta / 'asignacion_clusters.csv')
clave = ['id_jugador', 'season', 'team']

#%%
# 1. Coordenadas 2D (t-SNE), con caché
mapa = pd.read_csv(ruta_mapa) if ruta_mapa.exists() and not RECALCULAR else None
if mapa is None or len(mapa) != len(asig) or \
        not (mapa[clave].astype(str).values == asig[clave].astype(str).values).all():
    print("Calculando t-SNE (~1 min)...")
    Y = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto', random_state=0).fit_transform(X)
    mapa = asig[clave].copy()
    mapa['x'], mapa['y'] = Y[:, 0], Y[:, 1]
    mapa.to_csv(ruta_mapa, index=False)
else:
    print(f"Usando las coordenadas guardadas en {ruta_mapa.name}")
x, y = mapa['x'].to_numpy(), mapa['y'].to_numpy()

roles = asig.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index().str.split(': ').str[1]
K = len(roles)

#%%
# 2. Figura
n_col = 4
n_fil = int(np.ceil((K + 1) / n_col))
fig, axes = plt.subplots(n_fil, n_col, figsize=(17, 4.4 * n_fil))
axes = axes.ravel()
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

orden = np.random.default_rng(0).permutation(len(x))      # para que ningún grupo quede siempre arriba
for c, ax in zip(range(K), axes):
    en_rol = asig['Cluster'].to_numpy() == c
    ax.scatter(x[orden], y[orden], s=2, color=GRIS_PUNTO, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(x[en_rol], y[en_rol], s=3, color=AZUL, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_title(f"{roles[c]}  ·  n={en_rol.sum():,}".replace(',', '.'), fontsize=12)

# Panel de trayectorias
ax = axes[K]
ax.scatter(x, y, s=2, color=GRIS_PUNTO, alpha=0.5, linewidths=0, rasterized=True)
for c in range(K):        # nombre de cada rol donde más jugadores tiene, como referencia
    m = asig['Cluster'].to_numpy() == c
    h, bx, by = np.histogram2d(x[m], y[m], bins=12)
    i, j = np.unravel_index(h.argmax(), h.shape)
    ax.text((bx[i] + bx[i + 1]) / 2, (by[j] + by[j + 1]) / 2, roles[c].split(' / ')[0].replace('Delantero Finalizador (9)', 'Delantero 9'),
            ha='center', va='center', fontsize=8.5, color=TINTA_2, alpha=0.9)
JUGADORES = [('Trent Alexander-Arnold (1998)', 'Trent', AZUL, (8, -4), 'left'),
             ('Rodri (1996)', 'Rodri', NARANJA, (-8, 8), 'right'),
             ('Lionel Messi (1987)', 'Messi', AQUA, (8, -4), 'left')]
for id_j, corto, color, desplazamiento, alineacion in JUGADORES:
    filas = asig.index[asig['id_jugador'] == id_j]
    filas = filas[np.argsort(asig.loc[filas, 'season'].to_numpy())]
    ax.plot(x[filas], y[filas], color=color, linewidth=1.6, alpha=0.9, zorder=3)
    ax.scatter(x[filas], y[filas], s=36, color=color, edgecolor='white', linewidth=1.2, zorder=4)
    ult = filas[-1]
    temporada = str(asig.loc[ult, 'season'])
    primera = str(asig.loc[filas[0], 'season'])
    ax.annotate(f"{corto} ({primera[:2]}/{primera[2:]}→{temporada[:2]}/{temporada[2:]})", (x[ult], y[ult]),
                xytext=desplazamiento, textcoords='offset points', ha=alineacion,
                fontsize=10.5, fontweight='bold', color=TINTA, zorder=5)
ax.set_title('Trayectorias: Trent, Rodri y Messi', fontsize=12)
for ax in axes[K + 1:]:
    ax.axis('off')

fig.suptitle('Mapa de los 11.889 registros jugador-temporada (t-SNE): cada rol ocupa su propia zona',
             x=0.01, ha='left', fontsize=18, fontweight='bold')
fig.text(0.01, -0.01, 'Cada punto es un jugador en una temporada; los cercanos tienen perfiles parecidos. '
         'Los ejes y las distancias entre zonas lejanas no tienen significado en t-SNE.',
         fontsize=10.5, color=TINTA_2)
fig.tight_layout()
fig.savefig(carpeta_fig / 'R_mapa_2d_roles.png')
plt.show()
# %%
