#%%
# =====================================================================
#  CLASIFICADOR DE LA POSICIÓN NOMINAL (aprendizaje supervisado)
#
#  ¿Se puede adivinar si un jugador figura como DF, MF o FW solo con sus
#  55 métricas? Si sí, ¿en quiénes se equivoca el modelo?
#  Idea: los errores deberían caer justo en los jugadores híbridos y en los
#  roles de bordes difusos que encontró el clustering. Así la parte
#  supervisada valida a la no supervisada.
#
#  - Posición = la primera de pos_ (ej. "MF,DF" -> MF).
#  - Regresión logística multinomial sobre las métricas estandarizadas.
#  - Validación cruzada de 5 grupos AGRUPADA POR JUGADOR: todas las
#    temporadas de un jugador caen en el mismo grupo, así el modelo nunca
#    se evalúa con un jugador que ya vio en el entrenamiento.
#
#  Requiere Clustering/consenso.py (asignacion_clusters.csv).
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from comun import (CARPETA, CARPETA_FIG, AZUL, NARANJA, GRILLA, TINTA, TINTA_2, SECUENCIAL,
                   cargar, nombres_roles, abreviar)

asig, metricas, VARIABLES = cargar()
ROLES = nombres_roles(asig)
K = len(ROLES)

X = metricas[VARIABLES].to_numpy()
y = asig['pos_'].str.split(',').str[0].to_numpy()
hibrida = asig['pos_'].str.contains(',').to_numpy()
cv = GroupKFold(n_splits=5)

#%%
# 1. Validación cruzada agrupada por jugador
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
proba = cross_val_predict(lr, X, y, groups=asig['id_jugador'], cv=cv, method='predict_proba')
clases = np.sort(np.unique(y))
pred = clases[proba.argmax(axis=1)]
error = pred != y

gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
pred_gb = cross_val_predict(gb, X, y, groups=asig['id_jugador'], cv=cv)

print("=== ¿SE PUEDE ADIVINAR LA POSICIÓN NOMINAL CON LAS MÉTRICAS? ===")
print(f"Regresión logística : acierto {accuracy_score(y, pred):.1%} | acierto balanceado {balanced_accuracy_score(y, pred):.1%}")
print(f"Gradient boosting   : acierto {accuracy_score(y, pred_gb):.1%} | acierto balanceado {balanced_accuracy_score(y, pred_gb):.1%}")
print(f"Siempre 'DF' (base) : acierto {(y == 'DF').mean():.1%}")
confusion = pd.crosstab(pd.Series(y, name='Real'), pd.Series(pred, name='Predicha')) \
    .loc[['DF', 'MF', 'FW'], ['DF', 'MF', 'FW']]
print("\n", confusion.to_string())

#%%
# 2. ¿Dónde se equivoca?
print(f"\nError en posiciones puras (DF, MF, FW): {error[~hibrida].mean():.1%} (n={(~hibrida).sum()})")
print(f"Error en posiciones híbridas (ej. DF,MF): {error[hibrida].mean():.1%} (n={hibrida.sum()})")

por_rol = pd.DataFrame({'Rol': asig['Cluster'].map(ROLES), 'error': error, 'Confianza': asig['Confianza']}) \
    .groupby('Rol').agg(error=('error', 'mean'), confianza=('Confianza', 'mean'), n=('error', 'size')) \
    .loc[[ROLES[c] for c in range(K)]]
por_rol.to_csv(CARPETA / 'clasificador_error_por_rol.csv')
print("\n=== ERROR DEL CLASIFICADOR SEGÚN EL ROL DE CONSENSO ===")
print(por_rol.assign(error=lambda d: (d['error'] * 100).round(1), confianza=lambda d: d['confianza'].round(2)).to_string())
r = np.corrcoef(por_rol['error'], por_rol['confianza'])[0, 1]
print(f"Correlación entre error del clasificador y confianza del rol: {r:+.2f}")
print(f"Confianza media: aciertos {asig.loc[~error, 'Confianza'].mean():.2f} | errores {asig.loc[error, 'Confianza'].mean():.2f}")

# Jugadores "fuera de molde": posición pura, pero el modelo está muy seguro de que es otra
salida = asig[['id_jugador', 'season', 'team', 'league', 'pos_', 'Rol']].copy()
salida['Posicion'] = y
salida['Posicion_predicha'] = pred
for i, c in enumerate(clases):
    salida[f'Prob_{c}'] = proba[:, i]
salida['Prob_predicha'] = proba.max(axis=1)
salida.to_csv(CARPETA / 'clasificador_predicciones.csv', index=False)
molde = salida[error & ~hibrida].sort_values('Prob_predicha', ascending=False)
print("\n=== FUERA DE MOLDE: posición pura, pero sus números son de otra (prob. > 0.9) ===")
print(f"{(molde['Prob_predicha'] > 0.9).sum()} registros. Los más claros:")
print(molde.head(15)[['id_jugador', 'season', 'team', 'Posicion', 'Posicion_predicha', 'Prob_predicha', 'Rol']]
      .round(2).to_string(index=False))

#%%
# 3. Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [0.8, 1.2]})

pct = confusion.div(confusion.sum(axis=1), axis=0) * 100
ax1.imshow(pct.to_numpy(), cmap=SECUENCIAL, vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        v = pct.iat[i, j]
        ax1.text(j, i, f'{v:.0f}%\n(n={confusion.iat[i, j]:,})'.replace(',', '.'), ha='center', va='center',
                 fontsize=11, color='white' if v >= 50 else TINTA, fontweight='bold' if i == j else 'normal')
ax1.set_xticks(range(3), pct.columns)
ax1.set_yticks(range(3), pct.index)
ax1.set_xlabel('Posición predicha por el modelo')
ax1.set_ylabel('Posición nominal (FBref)')
ax1.tick_params(length=0)
ax1.spines[:].set_visible(False)
ax1.set_title(f'Acierta el {accuracy_score(y, pred):.0%} solo con las métricas')

orden = por_rol.sort_values('error')
yy = np.arange(len(orden))
colores = [NARANJA if e > 0.1 else AZUL for e in orden['error']]
ax2.barh(yy, orden['error'] * 100, color=colores, height=0.6)
for k, (rol, f) in enumerate(orden.iterrows()):
    ax2.text(f['error'] * 100 + 0.4, k, f"{f['error'] * 100:.1f}%   (confianza del rol {f['confianza']:.2f})",
             va='center', fontsize=10.5, color=TINTA_2)
ax2.set_yticks(yy, [abreviar(r).replace('\n', ' ') for r in orden.index])
ax2.set_xlim(0, 32)
ax2.set_xlabel('% de jugadores del rol cuya posición el modelo no acierta')
ax2.grid(axis='x', color=GRILLA)
ax2.set_axisbelow(True)
ax2.tick_params(length=0)
ax2.set_title('Se equivoca justo en los roles de bordes difusos')
fig.text(0.47, -0.02, f'Error en posiciones puras: {error[~hibrida].mean():.1%}  ·  en híbridas (ej. DF,MF): '
         f'{error[hibrida].mean():.1%}.  Validación cruzada agrupada por jugador.', fontsize=10.5, color=TINTA_2)

fig.tight_layout()
fig.savefig(CARPETA_FIG / 'O_clasificador_posicion.png')
plt.show()
# %%
