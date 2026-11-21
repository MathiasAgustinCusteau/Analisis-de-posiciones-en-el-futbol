#%%
# =====================================================================
#  MODELO FINAL: CLUSTERING DE CONSENSO (k = 7)
#  (evidence accumulation, Fred & Jain 2005)
#
#  Por qué: un único GMM con k=8 daba roles distintos según la semilla
#  (ver estabilidad_gmm.py). En vez de confiar en UNA corrida:
#
#   1. Se corre el GMM 200 veces, cada vez con una semilla distinta y con
#      el 80% de los registros elegidos al azar.
#   2. Para cada par de jugadores se cuenta en qué fracción de las corridas
#      cayeron en el mismo grupo  ->  matriz de co-asociación (0 a 1).
#   3. Los roles finales salen de agrupar esa matriz (enlace promedio):
#      quedan juntos los jugadores que el GMM junta "casi siempre".
#
#  Elección de k: PAC (Senbabaoglu et al., 2014) = proporción de pares
#  "ambiguos" (juntos entre el 10% y el 90% de las veces). Menor = mejor.
#  Reproducibilidad: se arma el consenso con las corridas 1-100 y con las
#  101-200 por separado y se comparan (ARI).
#
#  Probabilidades y outliers: con los grupos de consenso se estima un GMM
#  (peso, media y covarianza de cada rol). De ahí salen Prob_C* (las usa el
#  buscador de reemplazos) y Log_Densidad (detección de atípicos).
#
#  Orden del pipeline:
#   pca_global.py -> consenso.py -> clustering -> reemplazos.py / figuras_ppt.py
#
#  Salidas:
#   - asignacion_clusters.csv  : rol de cada registro (archivo oficial)
#   - consenso_k.csv           : PAC, reproducibilidad y confianza para cada k
#   - figuras/L_consenso.png
#  La primera vez guarda una copia del modelo viejo (k=8) en
#  asignacion_clusters_k8_original.csv.
#
#  Tarda ~5 min en 2 núcleos y usa ~2,5 GB de RAM.
# =====================================================================
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
carpeta_clu = RAIZ_PROYECTO / 'Clustering'
carpeta_fig = carpeta_clu / 'figuras'
carpeta_fig.mkdir(exist_ok=True)
ruta_asig = carpeta_clu / 'asignacion_clusters.csv'
ruta_vieja = carpeta_clu / 'asignacion_clusters_k8_original.csv'

KS = [6, 7, 8, 9]
K_ELEGIDO = 7         # el de menor PAC (ver consenso_k.csv)
N_CORRIDAS = 200
FRACCION = 0.8

# Firma de cada rol: métricas (z-score) que lo definen. Sirven para ponerle
# nombre a cada grupo de consenso sin depender del orden en que salen.
# El orden de la lista es "de atrás hacia adelante" y define la numeración C0..C6.
FIRMAS = {
    'Defensor Central':                ['despejes_p90', 'toques_area_propia_p90', 'tiros_bloqueados_p90'],
    'Lateral / Carrilero':             ['centros_area_rival_p90', 'quites_tercio_defensivo_p90', 'centros_totales_p90'],
    'Volante Recuperador':             ['quites_tercio_medio_p90', 'recuperaciones_p90', 'intercepciones_p90'],
    'Organizador / Regista':           ['pases_progresivos_p90', 'pases_tercio_ofensivo_p90', 'pases_cortos_completados_p90'],
    'Extremo / Creador':               ['regates_intentados_p90', 'acciones_creacion_tiro_p90', 'conducciones_progresivas_p90'],
    'Segunda Punta / Delantero Móvil': ['tiros_totales_p90', 'tiros_al_arco_p90', 'malos_controles_p90'],
    'Delantero Finalizador (9)':       ['fueras_de_juego_p90', 'npxg_p90', 'duelos_aereos_perdidos_p90'],
}

df_pca = pd.read_parquet(carpeta_clu / 'datos_pca.parquet')
X = df_pca.to_numpy()
meta = df_pca.index.to_frame(index=False)
n = len(X)
print(f"Matriz: {n} registros x {X.shape[1]} componentes")


def corrida(k, r):
    """Un GMM ajustado con una submuestra del 80%; asigna a TODOS los registros."""
    rng = np.random.default_rng(r)
    idx = rng.choice(n, int(FRACCION * n), replace=False)
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=1, random_state=r)
    return gmm.fit(X[idx]).predict(X)


def coasociacion(L, k):
    """Fracción de corridas en que cada par de registros quedó en el mismo grupo."""
    H = np.zeros((n, len(L) * k), dtype=np.float32)       # one-hot de todas las corridas
    for r, lbl in enumerate(L):
        H[np.arange(n), r * k + lbl] = 1
    return (H @ H.T) / len(L)


def consenso(C, k):
    """Agrupa la matriz de co-asociación con enlace promedio."""
    D = 1 - C
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1


def confianza(C, lbl):
    """Co-asociación media de cada registro con los demás miembros de su rol (0 a 1)."""
    out = np.zeros(n)
    for c in np.unique(lbl):
        idx = np.where(lbl == c)[0]
        out[idx] = C[np.ix_(idx, idx)].mean(axis=1)
    return out


def gmm_desde_grupos(X, lbl, k, reg=1e-6):
    """GMM cuyos parámetros (peso, media, covarianza) se estiman con los grupos de consenso."""
    gmm = GaussianMixture(n_components=k, covariance_type='full')
    pesos, medias, covs, prec_chol = [], [], [], []
    for c in range(k):
        Xc = X[lbl == c]
        cov = np.cov(Xc, rowvar=False, bias=True) + reg * np.eye(X.shape[1])
        pesos.append(len(Xc) / len(X))
        medias.append(Xc.mean(axis=0))
        covs.append(cov)
        prec_chol.append(solve_triangular(cholesky(cov, lower=True), np.eye(X.shape[1]), lower=True).T)
    gmm.weights_, gmm.means_, gmm.covariances_ = np.array(pesos), np.array(medias), np.array(covs)
    gmm.precisions_cholesky_ = np.array(prec_chol)
    gmm.precisions_ = np.array([p @ p.T for p in gmm.precisions_cholesky_])
    gmm.converged_, gmm.n_iter_, gmm.lower_bound_ = True, 0, np.nan
    return gmm


#%%
# 1. Corridas, PAC y reproducibilidad para cada k
filas, guardado = [], {}
for k in KS:
    print(f"\nk = {k}: {N_CORRIDAS} corridas de GMM...")
    L = np.array(Parallel(n_jobs=-1)(delayed(corrida)(k, r) for r in range(N_CORRIDAS)))

    mitad = N_CORRIDAS // 2
    lbl_a = consenso(coasociacion(L[:mitad], k), k)
    lbl_b = consenso(coasociacion(L[mitad:], k), k)

    C = coasociacion(L, k)
    v = C[np.triu_indices(n, 1)]
    pac = ((v > 0.1) & (v < 0.9)).mean()
    del v
    lbl = consenso(C, k)
    conf = confianza(C, lbl)
    filas.append({'k': k, 'PAC': pac,
                  'ARI_mitad1_vs_mitad2': adjusted_rand_score(lbl_a, lbl_b),
                  'Confianza_media': conf.mean()})
    print(pd.Series(filas[-1]).round(3).to_string())
    if k == K_ELEGIDO:
        guardado = {'C': C, 'lbl': lbl, 'conf': conf}
    del C

df_k = pd.DataFrame(filas)
df_k.to_csv(carpeta_clu / 'consenso_k.csv', index=False)
print("\n=== RESUMEN (PAC: menor es mejor | ARI mitades: 1 = el consenso es reproducible) ===")
print(df_k.round(3).to_string(index=False))

#%%
# 2. Nombres: cada grupo se empareja con la firma que mejor lo describe
lbl, conf = guardado['lbl'], guardado['conf']
df_orig = pd.read_csv(RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv')
metricas = [c for c in df_orig.columns if c not in meta.columns]
perfil_grupos = df_orig[metricas].groupby(lbl).mean()

nombres_firma = list(FIRMAS)
puntaje = np.array([[perfil_grupos.loc[g, FIRMAS[r]].mean() for r in nombres_firma]
                    for g in range(K_ELEGIDO)])
grupos_h, roles_h = linear_sum_assignment(-puntaje)
if K_ELEGIDO != len(FIRMAS):
    print(f"¡Ojo! Hay {len(FIRMAS)} firmas y {K_ELEGIDO} grupos: revisá los nombres a mano.")
# numeración C0..C(k-1) en el orden de FIRMAS (de atrás hacia adelante)
orden = sorted(zip(roles_h, grupos_h))
renum = {g: i for i, (_, g) in enumerate(orden)}
nombre = {g: nombres_firma[r] for r, g in orden}
print("\n=== PUNTAJE DE CADA GRUPO EN CADA FIRMA (z medio) ===")
print(pd.DataFrame(puntaje, columns=nombres_firma).round(2).to_string())

lbl_final = np.array([renum[g] for g in lbl])
nombres_clusters = {renum[g]: f"C{renum[g]}: {nombre[g]}" for g in range(K_ELEGIDO)}

# 3. Probabilidades y log-densidad
gmm = gmm_desde_grupos(X, lbl_final, K_ELEGIDO)
probas = gmm.predict_proba(X)
print(f"\nCoincidencia entre el rol de consenso y el más probable del GMM estimado: "
      f"{(probas.argmax(axis=1) == lbl_final).mean():.1%}")

res = meta.copy()
res['Cluster'] = lbl_final
res['Rol'] = res['Cluster'].map(nombres_clusters)
res['Prob_Max'] = probas.max(axis=1)
res['Log_Densidad'] = gmm.score_samples(X)
for c in range(K_ELEGIDO):
    res[f'Prob_C{c}'] = probas[:, c]
res['Confianza'] = conf

# Copia de seguridad del modelo viejo antes de pisarlo (solo la primera vez)
if ruta_asig.exists() and not ruta_vieja.exists():
    if pd.read_csv(ruta_asig, usecols=['Cluster'])['Cluster'].nunique() != K_ELEGIDO:
        shutil.copy(ruta_asig, ruta_vieja)
        print(f"Copia del modelo anterior guardada en {ruta_vieja.name}")
res.to_csv(ruta_asig, index=False)
print(f"Asignación guardada en {ruta_asig.name}")

#%%
# 4. Resumen
perfiles = df_orig[metricas].groupby(res['Rol'].values).mean()
print(f"\n=== PERFIL DE CADA ROL (k={K_ELEGIDO}, z-score medio) ===")
for rol in sorted(perfiles.index):
    s = perfiles.loc[rol].sort_values()
    mas = ', '.join(f"{i.replace('_p90', '')} ({v:+.2f})" for i, v in s[::-1][:4].items())
    menos = ', '.join(f"{i.replace('_p90', '')} ({v:+.2f})" for i, v in s[:2].items())
    m = res['Rol'] == rol
    print(f"{rol} (N={m.sum()}, confianza {res.loc[m, 'Confianza'].mean():.2f}): [+] {mas} | [-] {menos}")

if ruta_vieja.exists():
    vieja = pd.read_csv(ruta_vieja)
    print(f"\nARI con el modelo anterior (k=8): {adjusted_rand_score(res['Cluster'], vieja['Cluster']):.3f}")
    print(pd.crosstab(res['Rol'], vieja['Rol']).to_string())

#%%
# 5. Figura
AZUL, NARANJA, GRIS = '#2a78d6', '#eb6834', '#a3a29c'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1, 1.25]})

colores = [NARANJA if k == K_ELEGIDO else GRIS for k in df_k['k']]
ax1.bar(df_k['k'], df_k['PAC'], color=colores, width=0.6)
for k, pac, rep in zip(df_k['k'], df_k['PAC'], df_k['ARI_mitad1_vs_mitad2']):
    ax1.text(k, pac + 0.004, f"PAC {pac:.3f}\nreprod. {rep:.2f}", ha='center', va='bottom', fontsize=10)
ax1.set_xticks(KS)
ax1.set_ylim(0, df_k['PAC'].max() * 1.3)
ax1.set_xlabel('Número de roles (k)')
ax1.set_ylabel('PAC: pares de jugadores ambiguos')
ax1.set_title('¿Con cuántos roles hay menos ambigüedad?')

# Matriz de co-asociación ordenada por rol (muestra de 2500 registros para que se vea)
rng = np.random.default_rng(0)
muestra = np.sort(rng.choice(n, 2500, replace=False))
orden_fig = muestra[np.lexsort((-conf[muestra], lbl_final[muestra]))]
im = ax2.imshow(guardado['C'][np.ix_(orden_fig, orden_fig)], cmap='Blues', vmin=0, vmax=1,
                interpolation='nearest')
cl = lbl_final[orden_fig]
for b in np.flatnonzero(np.diff(cl)) + 0.5:
    ax2.axhline(b, color='white', lw=1)
    ax2.axvline(b, color='white', lw=1)
ax2.set_yticks([np.flatnonzero(cl == c).mean() for c in range(K_ELEGIDO)],
               [nombres_clusters[c].split(': ')[1] for c in range(K_ELEGIDO)], fontsize=10)
ax2.set_xticks([])
ax2.spines[:].set_visible(False)
ax2.set_title(f'Co-asociación entre jugadores (k={K_ELEGIDO}, {N_CORRIDAS} corridas)')
cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
cb.set_label('Fracción de corridas en el mismo rol')

fig.tight_layout()
fig.savefig(carpeta_fig / 'L_consenso.png', dpi=200, bbox_inches='tight')
plt.show()
del guardado
# %%
