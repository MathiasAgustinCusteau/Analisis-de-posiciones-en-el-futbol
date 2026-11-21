#%%
# =====================================================================
#  TRANSICIONES ENTRE ROLES Y PREDICCIÓN DEL ROL DE LA TEMPORADA SIGUIENTE
#
#  1. Matriz de transición (Markov): si un jugador tiene el rol A esta
#     temporada, ¿con qué probabilidad tiene el rol B la siguiente?
#  2. ¿Se puede predecir el rol del año siguiente mejor que "repite su rol"?
#     Regresión logística multinomial con: probabilidades de cada rol,
#     las 55 métricas (z), edad y liga. Se evalúa "hacia adelante": para
#     predecir la temporada T se entrena solo con transiciones anteriores a T.
#  3. Pronóstico 2025/26 para todos los jugadores de 2024/25.
#
#  Requiere Clustering/consenso.py (asignacion_clusters.csv).
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from comun import (CARPETA, CARPETA_FIG, NARANJA, GRIS, GRILLA, TINTA, SECUENCIAL,
                   cargar, nombres_roles, pares_consecutivos, abreviar)

asig, metricas, VARIABLES = cargar()
ROLES = nombres_roles(asig)
K = len(ROLES)
PROBS = [f'Prob_C{c}' for c in range(K)]
pares = pares_consecutivos(asig, metricas, VARIABLES)
print(f"Transiciones jugador-temporada -> temporada siguiente: {len(pares)}")

#%%
# 1. Matriz de transición y persistencia
matriz = pd.crosstab(pares['Cluster'], pares['Cluster_sig'], normalize='index')
matriz.index = matriz.columns = [ROLES[c] for c in range(K)]
matriz.to_csv(CARPETA / 'transiciones_matriz.csv')
print("\n=== MATRIZ DE TRANSICIÓN (fila = rol este año, columna = rol el año siguiente) ===")
print((matriz * 100).round(0).astype(int).to_string())

persistencia = 1 - pares['cambia'].mean()
reparto = asig['Cluster'].value_counts(normalize=True)
azar = (reparto ** 2).sum()   # prob. de repetir rol si se asignara al azar con las mismas proporciones
print(f"\nRepiten rol: {persistencia:.1%}  (al azar: {azar:.1%})")

cambia_liga = pares['league'] != pares['league_sig']
cambia_equipo = ~cambia_liga & (pares['team'] != pares['team_sig'])
mismo_equipo = pares['team'] == pares['team_sig']
print("\n=== ¿REPITE ROL? según el movimiento del jugador ===")
for nombre, m in [('Mismo equipo', mismo_equipo), ('Otro equipo, misma liga', cambia_equipo),
                  ('Otra liga', cambia_liga)]:
    print(f"  {nombre:25s} n={m.sum():5d}  repite {1 - pares.loc[m, 'cambia'].mean():.1%}")
print("  (Si los que cambian de liga repiten igual que los que cambian de equipo dentro de la\n"
      "   misma liga, la estandarización por liga hace comparables los roles entre ligas.)")

print("\n=== ¿REPITE ROL? según la confianza del consenso (quintiles) ===")
print((1 - pares.groupby(pd.qcut(pares['Confianza'], 5), observed=True)['cambia'].mean())
      .rename('repite').map('{:.1%}'.format).to_string())

#%%
# 2. Predicción del rol del año siguiente (evaluación hacia adelante)
ligas = pd.get_dummies(pares['league']).astype(float)
pares['edad2'] = pares['edad'] ** 2
X = pd.concat([pares[PROBS + ['edad', 'edad2'] + VARIABLES], ligas], axis=1).to_numpy()
y = pares['Cluster_sig'].to_numpy()


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))


filas = []
rng = np.random.default_rng(0)
for t in sorted(pares['season_sig'].unique())[2:]:          # las 2 primeras solo entrenan
    tr, te = (pares['season_sig'] < t).to_numpy(), (pares['season_sig'] == t).to_numpy()
    mdl = modelo().fit(X[tr], y[tr])
    pred, proba = mdl.predict(X[te]), mdl.predict_proba(X[te])
    repite = pares.loc[te, 'Cluster'].to_numpy()
    markov = pd.crosstab(pares.loc[tr, 'Cluster'], pares.loc[tr, 'Cluster_sig'], normalize='index') \
        .reindex(index=range(K), columns=range(K), fill_value=0).to_numpy()
    markov = np.clip(markov, 1e-6, 1)
    markov /= markov.sum(axis=1, keepdims=True)
    yt = y[te]
    cambian = repite != yt
    dif = (pred == yt).astype(float) - (repite == yt)
    boot = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
    filas.append({
        'temporada_predicha': t, 'n': te.sum(),
        'acierto_modelo': (pred == yt).mean(), 'acierto_repite_rol': (repite == yt).mean(),
        'mejora': dif.mean(), 'mejora_IC95_inf': np.percentile(boot, 2.5), 'mejora_IC95_sup': np.percentile(boot, 97.5),
        'cambios_anticipados': (pred[cambian] == yt[cambian]).mean(),
        'logloss_modelo': log_loss(yt, proba, labels=range(K)),
        'logloss_markov': log_loss(yt, markov[repite], labels=range(K)),
    })
res = pd.DataFrame(filas)
res.to_csv(CARPETA / 'prediccion_rol_por_temporada.csv', index=False)
print("\n=== PREDICCIÓN DEL ROL DEL AÑO SIGUIENTE (entrenando solo con el pasado) ===")
print(res.round(3).to_string(index=False))
peso = res['n'] / res['n'].sum()
print(f"\nPromedio: modelo {np.sum(peso * res['acierto_modelo']):.1%} vs repite rol "
      f"{np.sum(peso * res['acierto_repite_rol']):.1%}; anticipa el "
      f"{np.sum(peso * res['cambios_anticipados']):.0%} de los cambios de rol.")

#%%
# 3. Pronóstico 2025/26 (modelo entrenado con todas las transiciones)
final = modelo().fit(X, y)
ultima = asig['season'].max()
actual = asig[asig['season'] == ultima].drop_duplicates('id_jugador', keep=False).copy()
actual = actual.join(metricas.loc[actual.index, VARIABLES])
actual['edad'] += 1          # un año más
actual['edad2'] = actual['edad'] ** 2
lig = pd.get_dummies(actual['league']).reindex(columns=ligas.columns, fill_value=0).astype(float)
proba = final.predict_proba(pd.concat([actual[PROBS + ['edad', 'edad2'] + VARIABLES], lig], axis=1).to_numpy())
actual['Rol_actual'] = actual['Cluster'].map(ROLES)
actual['Rol_pronosticado'] = pd.Series(proba.argmax(axis=1), index=actual.index).map(ROLES)
actual['Prob_pronostico'] = proba.max(axis=1)
actual['Prob_cambiar_rol'] = 1 - proba[np.arange(len(actual)), actual['Cluster'].to_numpy()]
cols = ['id_jugador', 'team', 'league', 'pos_', 'edad', 'Rol_actual', 'Rol_pronosticado',
        'Prob_pronostico', 'Prob_cambiar_rol']
actual[cols].sort_values('Prob_cambiar_rol', ascending=False) \
    .to_csv(CARPETA / 'pronostico_rol_2526.csv', index=False)
print(f"\n=== PRONÓSTICO 2025/26: jugadores con mayor probabilidad de cambiar de rol ===")
print(actual.sort_values('Prob_cambiar_rol', ascending=False)[cols].head(15).round(2).to_string(index=False))

#%%
# 4. Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={'width_ratios': [1.25, 1]})

v = matriz.to_numpy() * 100
ax1.imshow(v, cmap=SECUENCIAL, vmin=0, vmax=90)
for i in range(K):
    for j in range(K):
        if v[i, j] >= 1:
            ax1.text(j, i, f'{v[i, j]:.0f}%', ha='center', va='center', fontsize=11,
                     color='white' if v[i, j] >= 45 else TINTA, fontweight='bold' if i == j else 'normal')
etiq = [abreviar(r) for r in matriz.index]
ax1.set_xticks(range(K), [e.replace('\n', ' ') for e in etiq], fontsize=11, rotation=35, ha='left')
ax1.set_yticks(range(K), [e.replace('\n', ' ') for e in etiq], fontsize=11)
ax1.xaxis.tick_top()
ax1.spines[:].set_visible(False)
ax1.set_xticks(np.arange(-.5, K), minor=True)
ax1.set_yticks(np.arange(-.5, K), minor=True)
ax1.grid(which='minor', color='white', linewidth=2)
ax1.tick_params(which='both', length=0)
ax1.set_xlabel('Rol la temporada siguiente')
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Rol esta temporada')
ax1.set_title(f'¿Qué rol tiene el año siguiente? (repite el {persistencia:.0%}; al azar, {azar:.0%})\n', pad=4)

x = res['temporada_predicha'].astype(int).map(lambda s: f"{str(s)[:2]}/{str(s)[2:]}")
ax2.plot(x, res['acierto_repite_rol'] * 100, marker='o', color=GRIS, lw=2, ms=8, label='Repite su rol (base)')
ax2.plot(x, res['acierto_modelo'] * 100, marker='o', color=NARANJA, lw=2.5, ms=9, label='Modelo (regresión logística)')
for xi, r in zip(x, res.itertuples()):
    ax2.text(xi, r.acierto_modelo * 100 + 0.8, f'+{r.mejora * 100:.1f}', ha='center', fontsize=10, color=NARANJA)
ax2.set_ylim(55, 80)
ax2.set_ylabel('Acierto en el rol del año siguiente (%)')
ax2.set_xlabel('Temporada predicha (entrenando solo con temporadas anteriores)')
ax2.grid(axis='y', color=GRILLA)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc='lower left')
ax2.set_title('¿Se puede anticipar el rol del año siguiente?')

fig.tight_layout()
fig.savefig(CARPETA_FIG / 'M_transiciones_roles.png')
plt.show()
# %%
