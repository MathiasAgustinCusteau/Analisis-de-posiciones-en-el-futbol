#%%
# =====================================================================
#  ESTABILIDAD DE LOS ROLES (GMM)
#  ¿Los 8 roles son reales o dependen de la semilla / de los datos?
#
#  Dos pruebas, para k = 6, 7, 8 y 9, usando el MISMO procedimiento que
#  el modelo final (GMM covarianza completa, n_init=15):
#
#   1. Semillas: se ajusta el modelo con 10 semillas distintas y se mide
#      cuánto coinciden las asignaciones entre sí (ARI promedio de a pares).
#   2. Submuestras: se ajusta el modelo con el 80% de los registros elegidos
#      al azar (20 veces), se asigna a TODOS los jugadores y se compara con
#      el modelo ajustado con el 100% (ARI).
#
#  Para k = 8 se mide además la estabilidad de CADA rol (índice de Jaccard
#  entre el rol original y el rol más parecido en cada submuestra).
#  Criterio usual (Hennig, 2007): Jaccard medio > 0.75 = rol estable;
#  0.6–0.75 = hay un patrón pero con bordes difusos; < 0.5 = el rol se disuelve.
#
#  Evalúa el modelo ORIGINAL (un único GMM con k=8, semilla 42). Su resultado
#  es lo que motivó pasar al consenso con k=7 (ver consenso.py).
#  Requiere asignacion_clusters_k8_original.csv (lo guarda consenso.py).
#  Tarda varios minutos (~150 ajustes de GMM); usa todos los núcleos.
# =====================================================================
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
carpeta_clu = RAIZ_PROYECTO / 'Clustering'
carpeta_fig = carpeta_clu / 'figuras'
carpeta_fig.mkdir(exist_ok=True)

KS = [6, 7, 8, 9]
K_ELEGIDO = 8        # el modelo original que se evalúa
N_SEMILLAS = 10
N_SUBMUESTRAS = 20
FRACCION = 0.8
N_INIT = 15          # igual que el modelo final de Clustering/clustering

X = pd.read_parquet(carpeta_clu / 'datos_pca.parquet').to_numpy()
asig = pd.read_csv(carpeta_clu / 'asignacion_clusters_k8_original.csv')
ref = asig['Cluster'].to_numpy()
nombres = asig.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index()
print(f"Matriz: {X.shape[0]} registros x {X.shape[1]} componentes")


def ajustar(k, semilla, idx=None):
    """Ajusta un GMM (con todos los datos o con la submuestra idx) y asigna a TODOS."""
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=N_INIT, random_state=semilla)
    gmm.fit(X if idx is None else X[idx])
    return gmm.predict(X)


def jaccard_por_cluster(lbl_ref, lbl_new, k):
    """Para cada clúster de referencia, Jaccard con el clúster nuevo más parecido."""
    out = []
    for c in range(k):
        a = lbl_ref == c
        mejor = max(((a & (lbl_new == j)).sum() / (a | (lbl_new == j)).sum()) for j in np.unique(lbl_new))
        out.append(mejor)
    return out


rng = np.random.default_rng(2026)
subm = [rng.choice(len(X), int(FRACCION * len(X)), replace=False) for _ in range(N_SUBMUESTRAS)]

#%%
# 1. Ajustes (en paralelo)
tareas = []
for k in KS:
    tareas += [('completo', k, 42, None)]
    tareas += [('semilla', k, s, None) for s in range(N_SEMILLAS)]
    tareas += [('submuestra', k, 1000 + r, subm[r]) for r in range(N_SUBMUESTRAS)]

print(f"Ajustando {len(tareas)} modelos GMM...")
etiquetas = Parallel(n_jobs=-1, verbose=5)(delayed(ajustar)(k, s, idx) for _, k, s, idx in tareas)
res = {}
for (tipo, k, s, _), lbl in zip(tareas, etiquetas):
    res.setdefault((tipo, k), []).append(lbl)

#%%
# 2. Resumen por k
filas = []
for k in KS:
    completo = res[('completo', k)][0]
    ari_sem = [adjusted_rand_score(a, b) for a, b in combinations(res[('semilla', k)], 2)]
    ari_sub = [adjusted_rand_score(completo, l) for l in res[('submuestra', k)]]
    filas.append({
        'k': k,
        'ARI_semillas_media': np.mean(ari_sem), 'ARI_semillas_min': np.min(ari_sem),
        'ARI_submuestras_media': np.mean(ari_sub), 'ARI_submuestras_min': np.min(ari_sub),
    })
df_k = pd.DataFrame(filas)
df_k.to_csv(carpeta_clu / 'estabilidad_k.csv', index=False)
print("\n=== ESTABILIDAD POR k (ARI: 1 = asignaciones idénticas, 0 = azar) ===")
print(df_k.round(3).to_string(index=False))

# Chequeo: el modelo completo con k=8 tiene que reproducir el modelo original
print(f"\nARI del modelo k={K_ELEGIDO} vs el modelo original: "
      f"{adjusted_rand_score(res[('completo', K_ELEGIDO)][0], ref):.3f}")

# 3. Estabilidad de cada rol (k = 8)
jac_sub = np.array([jaccard_por_cluster(ref, l, K_ELEGIDO) for l in res[('submuestra', K_ELEGIDO)]])
jac_sem = np.array([jaccard_por_cluster(ref, l, K_ELEGIDO) for l in res[('semilla', K_ELEGIDO)]])
df_c = pd.DataFrame({
    'Cluster': range(K_ELEGIDO),
    'Rol': nombres.values,
    'N': np.bincount(ref, minlength=K_ELEGIDO),
    'Jaccard_submuestras_media': jac_sub.mean(axis=0),
    'Jaccard_submuestras_min': jac_sub.min(axis=0),
    'Jaccard_semillas_media': jac_sem.mean(axis=0),
})
df_c['Veredicto'] = pd.cut(df_c['Jaccard_submuestras_media'], [0, 0.5, 0.6, 0.75, 1.01],
                           labels=['Se disuelve', 'Débil', 'Patrón difuso', 'Estable'], right=False)
df_c.to_csv(carpeta_clu / 'estabilidad_roles_k8.csv', index=False)
print(f"\n=== ESTABILIDAD DE CADA ROL (k={K_ELEGIDO}, Jaccard) ===")
print(df_c.round(3).to_string(index=False))

#%%
# 4. Figura
AZUL, NARANJA, GRIS = '#2a78d6', '#eb6834', '#a3a29c'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})

for col, color, etiq, dx in [('semillas', AZUL, 'Distintas semillas', -0.08),
                             ('submuestras', NARANJA, 'Submuestras del 80%', 0.08)]:
    ax1.errorbar(df_k['k'] + dx, df_k[f'ARI_{col}_media'],
                 yerr=[df_k[f'ARI_{col}_media'] - df_k[f'ARI_{col}_min'], np.zeros(len(df_k))],
                 fmt='o', ms=9, capsize=0, color=color, label=etiq)
ax1.set_xticks(KS)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel('Número de roles (k)')
ax1.set_ylabel('ARI (media; la línea baja hasta el mínimo)')
ax1.set_title('¿Se repiten las asignaciones?')
ax1.axvspan(K_ELEGIDO - 0.4, K_ELEGIDO + 0.4, color=GRIS, alpha=0.15, lw=0)
ax1.legend(frameon=False, loc='lower left')

orden = df_c.sort_values('Jaccard_submuestras_media')
y = np.arange(len(orden))
ax2.hlines(y, orden['Jaccard_submuestras_min'], orden['Jaccard_submuestras_media'], color='#d8d7d2', lw=5)
ax2.scatter(orden['Jaccard_submuestras_media'], y, s=110, color=NARANJA, zorder=3, label='Media (20 submuestras)')
ax2.scatter(orden['Jaccard_submuestras_min'], y, s=160, color='#52514e',
            marker='|', zorder=3, label='Peor submuestra')
ax2.axvline(0.75, color=GRIS, ls='--', lw=1)
ax2.text(0.76, -0.45, 'estable\n(> 0.75)', color='#52514e', fontsize=10, va='bottom')
ax2.set_yticks(y, [f"{r.split(': ')[1]} (n={n})" for r, n in zip(orden['Rol'], orden['N'])])
ax2.set_xlim(0, 1.0)
ax2.set_xlabel('Jaccard con el rol original (1 = mismo grupo de jugadores)')
ax2.set_title(f'¿Qué roles sobreviven al cambiar los datos? (k={K_ELEGIDO})')
ax2.legend(frameon=False, loc='upper left')

fig.tight_layout()
fig.savefig(carpeta_fig / 'K_estabilidad_gmm.png', dpi=200, bbox_inches='tight')
plt.show()
# %%
