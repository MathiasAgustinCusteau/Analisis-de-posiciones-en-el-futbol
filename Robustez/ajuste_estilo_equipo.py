#%%
# =====================================================================
#  AJUSTE POR ESTILO DE EQUIPO (análisis de robustez)
#
#  Problema: un jugador de un equipo que tiene mucho la pelota (City,
#  Barcelona) acumula más pases y toques que uno de un equipo que la cede,
#  aunque cumplan la misma función. ¿Los roles reflejan al jugador o a su equipo?
#
#  1. Índice de estilo de cada equipo-temporada: pases por partido del equipo
#     (sumando a TODOS sus jugadores, también los de <900 min) respecto del
#     promedio de su liga y temporada, en escala log. Es una aproximación a la
#     posesión (FBref la calcula justamente como % de pases del partido).
#  2. Se le quita a cada métrica la parte que explica el estilo del equipo
#     (regresión lineal: métrica ~ índice de estilo; se usa el residuo) y se
#     vuelve a estandarizar por liga y temporada.
#  3. Con la matriz ajustada se repite el pipeline (PCA 90% + consenso de GMM)
#     y se compara con los roles oficiales.
#
#  No reemplaza al modelo oficial: mide cuánto dependen los roles del estilo.
#  Tarda ~4 min en 2 núcleos.
# =====================================================================
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA = RAIZ_PROYECTO / 'Robustez'
CARPETA_FIG = CARPETA / 'figuras'
CARPETA_FIG.mkdir(parents=True, exist_ok=True)
META = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']
KS = [6, 7, 8, 9]
K_OFICIAL = 7
N_CORRIDAS = 100
FRACCION = 0.8

AZUL, NARANJA, GRIS, GRILLA, TINTA_2 = '#2a78d6', '#eb6834', '#a3a29c', '#e4e3df', '#52514e'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': GRILLA, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

#%%
# 1. Índice de estilo de cada equipo-temporada
equipos = []
for ruta in sorted(glob.glob(str(RAIZ_PROYECTO / 'datos_kaggle' / 'ligas_temporadas' / '*.csv'))):
    d = pd.read_csv(ruta)
    for c in ['Total_Att', 'Playing Time_90s']:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    g = d.groupby(['league', 'season', 'team'], as_index=False).agg(
        pases=('Total_Att', 'sum'), noventas=('Playing Time_90s', 'sum'))
    g['partidos'] = g['noventas'] / 11            # 11 jugadores x 90 min = 1 partido
    g['pases_por_partido'] = g['pases'] / g['partidos']
    equipos.append(g)
equipos = pd.concat(equipos, ignore_index=True)
equipos['indice_estilo'] = np.log(equipos['pases_por_partido'] /
                                  equipos.groupby(['league', 'season'])['pases_por_partido'].transform('mean'))
equipos.to_csv(CARPETA / 'indice_estilo_equipos.csv', index=False)
print("=== EQUIPOS CON MÁS Y MENOS PELOTA (índice de estilo) ===")
orden_eq = equipos.sort_values('indice_estilo')
print(pd.concat([orden_eq.tail(5), orden_eq.head(5)])[['league', 'season', 'team', 'pases_por_partido', 'indice_estilo']]
      .round(2).to_string(index=False))

#%%
# 2. Métricas ajustadas por estilo
metricas = pd.read_csv(RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv')
asig = pd.read_csv(RAIZ_PROYECTO / 'Clustering' / 'asignacion_clusters.csv')
VARIABLES = [c for c in metricas.columns if c not in META]
datos = metricas.merge(equipos[['league', 'season', 'team', 'indice_estilo']],
                       on=['league', 'season', 'team'], how='left', validate='many_to_one')
if datos['indice_estilo'].isna().any():
    raise ValueError("Hay jugadores sin índice de estilo de su equipo.")
estilo = datos['indice_estilo'].to_numpy()

ajustada = datos[VARIABLES].copy()
r2 = {}
for v in VARIABLES:
    pendiente, ordenada = np.polyfit(estilo, ajustada[v], 1)
    residuo = ajustada[v] - (pendiente * estilo + ordenada)
    r2[v] = 1 - residuo.var() / ajustada[v].var()
    ajustada[v] = residuo
ajustada = ajustada.groupby([datos['league'], datos['season']]).transform(lambda s: (s - s.mean()) / s.std())
r2 = pd.Series(r2, name='R2_estilo').sort_values(ascending=False)
r2.to_csv(CARPETA / 'r2_estilo_por_metrica.csv')
print(f"\n=== CUÁNTO DE CADA MÉTRICA EXPLICA EL ESTILO DEL EQUIPO (R²) ===")
print(r2.head(10).round(3).to_string())
print(f"Mediana en las {len(VARIABLES)} métricas: {r2.median():.3f}")

pca = PCA().fit(ajustada)
n_comp = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1)
X = PCA(n_components=n_comp).fit_transform(ajustada)
n = len(X)
print(f"\nPCA de la matriz ajustada: {n_comp} componentes para el 90% de la varianza")

#%%
# 3. Consenso con la matriz ajustada (mismo método que Clustering/consenso.py)
def corrida(k, r):
    idx = np.random.default_rng(r).choice(n, int(FRACCION * n), replace=False)
    return GaussianMixture(n_components=k, covariance_type='full', n_init=1, random_state=r).fit(X[idx]).predict(X)


def consenso(L, k):
    H = np.zeros((n, len(L) * k), dtype=np.float32)
    for r, lbl in enumerate(L):
        H[np.arange(n), r * k + lbl] = 1
    C = (H @ H.T) / len(L)
    v = C[np.triu_indices(n, 1)]
    pac = ((v > 0.1) & (v < 0.9)).mean()
    del v
    D = 1 - C
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), 'average'), k, 'maxclust') - 1, pac


filas, lbl_ajustado = [], None
for k in KS:
    L = np.array(Parallel(n_jobs=-1)(delayed(corrida)(k, r) for r in range(N_CORRIDAS)))
    lbl, pac = consenso(L, k)
    filas.append({'k': k, 'PAC_ajustado': pac, 'ARI_vs_oficial': adjusted_rand_score(lbl, asig['Cluster'])})
    print(f"k={k}: PAC {pac:.3f} | ARI con los roles oficiales {filas[-1]['ARI_vs_oficial']:.3f}")
    if k == K_OFICIAL:
        lbl_ajustado = lbl
pd.DataFrame(filas).to_csv(CARPETA / 'ajuste_estilo_por_k.csv', index=False)

# Emparejar cada grupo ajustado con el rol oficial que más jugadores comparte
tabla = pd.crosstab(lbl_ajustado, asig['Cluster'])
f, c = linear_sum_assignment(-tabla.values)
pareja = {tabla.index[i]: tabla.columns[j] for i, j in zip(f, c)}
roles = asig.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index()

comp = asig[['id_jugador', 'season', 'team', 'league', 'pos_', 'Cluster', 'Rol']].copy()
comp['Cluster_ajustado'] = [pareja[g] for g in lbl_ajustado]
comp['Rol_ajustado'] = comp['Cluster_ajustado'].map(roles)
comp['indice_estilo'] = estilo
comp['cambia'] = comp['Cluster'] != comp['Cluster_ajustado']
comp.to_csv(CARPETA / 'asignacion_ajuste_estilo.csv', index=False)

print(f"\n=== ROLES OFICIALES vs AJUSTADOS POR ESTILO (k={K_OFICIAL}) ===")
print(f"Mantienen el rol: {1 - comp['cambia'].mean():.1%}")
print(pd.crosstab(comp['Rol'], comp['Rol_ajustado']).to_string())

comp['quintil_estilo'] = pd.qcut(comp['indice_estilo'], 5,
                                 labels=['Muy poca pelota', 'Poca', 'Media', 'Mucha', 'Muchísima'])
por_quintil = comp.groupby('quintil_estilo', observed=True)['cambia'].mean()
print("\n% que cambia de rol según cuánto tiene la pelota su equipo:")
print((por_quintil * 100).round(1).to_string())

por_rol = comp.groupby('Rol')['cambia'].mean().sort_values()
print("\n% que cambia de rol, por rol oficial:")
print((por_rol * 100).round(1).to_string())

print("\n=== CASOS DE REFERENCIA ===")
ref = ['Trent Alexander-Arnold (1998)', 'Rodri (1996)', 'Lionel Messi (1987)', 'Kevin De Bruyne (1991)']
print(comp[comp['id_jugador'].isin(ref)].sort_values(['id_jugador', 'season'])
      [['id_jugador', 'season', 'team', 'Rol', 'Rol_ajustado']].to_string(index=False))

#%%
# 4. Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.1, 1]})

top = r2.head(10)[::-1]
ax1.barh(range(len(top)), top.values * 100, color=AZUL, height=0.62)
for i, v in enumerate(top.values):
    ax1.text(v * 100 + 0.6, i, f'{v * 100:.0f}%', va='center', fontsize=11, color=TINTA_2)
ax1.set_yticks(range(len(top)), [t.replace('_p90', '').replace('_', ' ') for t in top.index])
ax1.set_xlabel('% de la variación de la métrica que explica el estilo del equipo (R²)')
ax1.set_xlim(0, 55)
ax1.grid(axis='x', color=GRILLA)
ax1.set_axisbelow(True)
ax1.tick_params(length=0)
ax1.set_title(f'El estilo pesa en los pases (mediana de las {len(VARIABLES)} métricas: {r2.median():.0%})')

colores = [GRIS] * 4 + [NARANJA]
ax2.bar(range(5), por_quintil.values * 100, color=colores, width=0.62)
for i, v in enumerate(por_quintil.values):
    ax2.text(i, v * 100 + 0.3, f'{v * 100:.0f}%', ha='center', fontsize=11, color=TINTA_2)
ax2.set_xticks(range(5), [str(q).replace(' ', '\n', 1) if q == 'Muy poca pelota' else q for q in por_quintil.index])
ax2.set_xlabel('Cuánto tiene la pelota el equipo del jugador (quintiles del índice de estilo)')
ax2.set_ylabel('% que cambia de rol al ajustar por estilo')
ax2.set_ylim(0, max(por_quintil.values) * 100 * 1.3)
ax2.grid(axis='y', color=GRILLA)
ax2.set_axisbelow(True)
ax2.set_title(f'Pero el {1 - comp["cambia"].mean():.0%} de los jugadores mantiene su rol')

fig.tight_layout()
fig.savefig(CARPETA_FIG / 'P_ajuste_estilo_equipo.png')
plt.show()
# %%
