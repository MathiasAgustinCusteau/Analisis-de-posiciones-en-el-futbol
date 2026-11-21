#%%
# =====================================================================
#  EDAD Y CAMBIO DE ROL
#
#  1. ¿Cómo cambia el reparto de roles con la edad? (todos los registros)
#     Ojo: mezcla dos efectos. Los jugadores cambian de rol, pero además
#     solo "sobreviven" hasta los 33 algunos perfiles (≥900 min).
#  2. Para separar esos efectos: los MISMOS jugadores, comparando su primer
#     registro con el último, con al menos 4 años de diferencia.
#  3. ¿La probabilidad de cambiar de rol de un año al otro depende de la
#     edad? Regresión logística: cambia ~ edad + edad² + rol + confianza + liga.
#     El efecto de la edad se muestra como probabilidad predicha a cada edad
#     (con el resto de las variables como están), con IC 95% por bootstrap.
#
#  Requiere Clustering/consenso.py (asignacion_clusters.csv).
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from comun import (CARPETA, CARPETA_FIG, AZUL, NARANJA, GRIS, GRILLA, TINTA_2,
                   cargar, nombres_roles, pares_consecutivos, abreviar)

asig, _, _ = cargar()
ROLES = nombres_roles(asig)
K = len(ROLES)

#%%
# 1. Reparto de roles por grupo de edad (todos los registros)
reparto = pd.crosstab(asig['Cluster'].map(ROLES), asig['grupo_edad'], normalize='columns')
reparto = reparto.loc[[ROLES[c] for c in range(K)]]
print("=== % DE CADA ROL SEGÚN LA EDAD (todos los registros) ===")
print((reparto * 100).round(1).to_string())
print("N por grupo:", asig['grupo_edad'].value_counts().sort_index().to_dict())

#%%
# 2. Mismos jugadores: primer registro vs último (≥ 4 años después)
orden = asig.sort_values(['id_jugador', 'edad'])
primero = orden.groupby('id_jugador').first()
ultimo = orden.groupby('id_jugador').last()
ok = (ultimo['edad'] - primero['edad']) >= 4
mismos = pd.DataFrame({
    'joven': primero.loc[ok, 'Cluster'].map(ROLES).value_counts(normalize=True),
    'mayor': ultimo.loc[ok, 'Cluster'].map(ROLES).value_counts(normalize=True),
}).reindex([ROLES[c] for c in range(K)])
mismos['cambio_pp'] = (mismos['mayor'] - mismos['joven']) * 100
mismos.to_csv(CARPETA / 'edad_mismos_jugadores.csv')
print(f"\n=== MISMOS JUGADORES ({ok.sum()}), primer registro vs último (≥4 años después) ===")
print(f"Edad media: {primero.loc[ok, 'edad'].mean():.1f} -> {ultimo.loc[ok, 'edad'].mean():.1f}")
print((mismos * [100, 100, 1]).round(1).to_string())

destino = pd.crosstab(primero.loc[ok, 'Cluster'].map(ROLES), ultimo.loc[ok, 'Cluster'].map(ROLES),
                      normalize='index')
print("\n¿Dónde terminan los que empezaron como Extremo / Creador?")
print((destino.loc[ROLES[4]] * 100).round(0).sort_values(ascending=False).astype(int).to_string())

#%%
# 3. Probabilidad de cambiar de rol según la edad
pares = pares_consecutivos(asig)
por_edad = pares.groupby('grupo_edad', observed=True)['cambia'].agg(['mean', 'size'])
print("\n=== % QUE CAMBIA DE ROL AL AÑO SIGUIENTE, POR EDAD (observado) ===")
print(por_edad.assign(mean=lambda d: (d['mean'] * 100).round(1)).to_string())


def matriz_X(df, edad=None):
    e = df['edad'] if edad is None else pd.Series(edad, index=df.index)
    return np.column_stack([
        (e - 26) / 4, ((e - 26) / 4) ** 2,
        pd.get_dummies(df['Cluster']).reindex(columns=range(K), fill_value=0).iloc[:, 1:].astype(float),
        df['Confianza'],
        pd.get_dummies(df['league']).iloc[:, 1:].astype(float),
    ])


EDADES = np.arange(19, 36)


def curva(df):
    mdl = LogisticRegression(C=1e4, max_iter=5000).fit(matriz_X(df), df['cambia'])
    return np.array([mdl.predict_proba(matriz_X(df, e))[:, 1].mean() for e in EDADES])


pred = curva(pares)
rng = np.random.default_rng(0)
boot = np.array([curva(pares.iloc[rng.integers(0, len(pares), len(pares))]) for _ in range(200)])
ic_inf, ic_sup = np.percentile(boot, [2.5, 97.5], axis=0)
efecto = pd.DataFrame({'edad': EDADES, 'prob_cambiar': pred, 'IC95_inf': ic_inf, 'IC95_sup': ic_sup})
efecto.to_csv(CARPETA / 'edad_prob_cambiar_rol.csv', index=False)
print("\n=== PROB. PREDICHA DE CAMBIAR DE ROL (mismo rol, confianza y liga; solo cambia la edad) ===")
print(efecto[efecto['edad'].isin([19, 22, 25, 28, 31, 34])].round(3).to_string(index=False))
dif = boot[:, EDADES == 31][:, 0] - boot[:, EDADES == 22][:, 0]
print(f"Diferencia 31 vs 22 años: {pred[EDADES == 31][0] - pred[EDADES == 22][0]:+.3f} "
      f"(IC95 {np.percentile(dif, 2.5):+.3f} a {np.percentile(dif, 97.5):+.3f})")

#%%
# 4. Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={'width_ratios': [1, 1.1]})

ax1.fill_between(EDADES, ic_inf * 100, ic_sup * 100, color=NARANJA, alpha=0.15, lw=0)
ax1.plot(EDADES, pred * 100, color=NARANJA, lw=2.5, label='Modelo (a igual rol, confianza y liga)')
centros = [19.5, 23, 26, 29, 32, 35]
ax1.scatter(centros, por_edad['mean'] * 100, s=70, color=GRIS, zorder=3, label='Observado por grupo de edad')
ax1.set_xlabel('Edad al inicio de la temporada')
ax1.set_ylabel('Probabilidad de cambiar de rol al año siguiente (%)')
ax1.set_ylim(0, 50)
ax1.grid(axis='y', color=GRILLA)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc='lower left')
ax1.set_title('Los jóvenes cambian más de rol')

orden_r = mismos.sort_values('cambio_pp').index
y = np.arange(len(orden_r))
for yi, r in zip(y, orden_r):
    j, m = mismos.loc[r, 'joven'] * 100, mismos.loc[r, 'mayor'] * 100
    ax2.annotate('', xy=(m, yi), xytext=(j, yi),
                 arrowprops=dict(arrowstyle='->', color=NARANJA if m > j else AZUL, lw=2))
    ax2.text(max(j, m) + 0.6, yi, f"{m - j:+.1f} pp", va='center', fontsize=10.5, color=TINTA_2)
ax2.scatter(mismos.loc[orden_r, 'joven'] * 100, y, s=60, color=GRIS, zorder=3, label='Primer registro (joven)')
ax2.set_yticks(y, [abreviar(r).replace('\n', ' ') for r in orden_r])
ax2.set_xlabel('% de los jugadores en ese rol')
ax2.set_xlim(5, 27)
ax2.grid(axis='x', color=GRILLA)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc='upper right', title='La flecha va del primer registro al último', title_fontsize=10.5)
ax2.set_title(f'Los mismos {ok.sum()} jugadores, 4+ años después')

fig.tight_layout()
fig.savefig(CARPETA_FIG / 'N_edad_y_roles.png')
plt.show()
# %%
