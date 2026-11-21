"""Carga de datos y estilo compartidos por los scripts de Prediccion/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA = RAIZ_PROYECTO / 'Prediccion'
CARPETA_FIG = CARPETA / 'figuras'
CARPETA_FIG.mkdir(parents=True, exist_ok=True)

META = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']
GRUPOS_EDAD = ([15, 21, 24, 27, 30, 33, 45], ['≤21', '22-24', '25-27', '28-30', '31-33', '34+'])

# --- Estilo (el mismo de Clustering/figuras_ppt.py) ---
TINTA, TINTA_2, GRILLA = '#0b0b0b', '#52514e', '#e4e3df'
AZUL, NARANJA, AQUA, GRIS = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29c'
SECUENCIAL = LinearSegmentedColormap.from_list(
    'azules', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA,
    'axes.edgecolor': GRILLA, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def cargar():
    """Devuelve (asig, metricas, VARIABLES): roles de consenso + métricas z, alineados fila a fila."""
    asig = pd.read_csv(RAIZ_PROYECTO / 'Clustering' / 'asignacion_clusters.csv')
    metricas = pd.read_csv(RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv')
    clave = ['id_jugador', 'season', 'team']
    if not (asig[clave].astype(str).values == metricas[clave].astype(str).values).all():
        raise ValueError("asignacion_clusters.csv y matriz_master_pca.csv no están alineadas: "
                         "corré Clustering/consenso.py.")
    # Edad al inicio de la temporada (ej. temporada 2324 -> 2023 - año de nacimiento)
    nacimiento = asig['id_jugador'].str.extract(r'\((\d{4})\)$')[0].astype(int)
    asig['edad'] = 2000 + asig['season'] // 100 - nacimiento
    asig['grupo_edad'] = pd.cut(asig['edad'], bins=GRUPOS_EDAD[0], labels=GRUPOS_EDAD[1])
    variables = [c for c in metricas.columns if c not in META]
    return asig, metricas, variables


def nombres_roles(asig):
    """{número de rol: nombre corto}"""
    roles = asig.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index()
    return {c: r.split(': ', 1)[1] for c, r in roles.items()}


def pares_consecutivos(asig, metricas=None, variables=None):
    """Una fila por jugador observado en dos temporadas seguidas: rol en t y en t+1.
    Se excluyen las temporadas en que el jugador pasó por dos equipos (no hay una sola fila)."""
    unico = asig.drop_duplicates(['id_jugador', 'season'], keep=False)
    if metricas is not None:
        unico = unico.join(metricas.loc[unico.index, variables])
    temporadas = sorted(asig['season'].unique())
    siguiente = dict(zip(temporadas[:-1], temporadas[1:]))
    t0 = unico.assign(season_sig=unico['season'].map(siguiente))
    t1 = unico[['id_jugador', 'season', 'team', 'league', 'Cluster']].rename(columns={
        'season': 'season_sig', 'team': 'team_sig', 'league': 'league_sig', 'Cluster': 'Cluster_sig'})
    pares = t0.merge(t1, on=['id_jugador', 'season_sig'])
    pares['cambia'] = (pares['Cluster'] != pares['Cluster_sig']).astype(int)
    return pares


ABREVIADO = {'Defensor Central': 'Central', 'Lateral / Carrilero': 'Lateral', 'Volante Recuperador': 'Volante',
             'Organizador / Regista': 'Organizador', 'Extremo / Creador': 'Extremo',
             'Segunda Punta / Delantero Móvil': 'Segunda\npunta', 'Delantero Finalizador (9)': 'Delantero 9'}


def abreviar(nombre):
    """Nombre corto de un rol para ejes con poco lugar."""
    return ABREVIADO.get(nombre, nombre.split(' / ')[0])
