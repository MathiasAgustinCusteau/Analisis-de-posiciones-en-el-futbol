#%%
# =====================================================================
#  BUSCADOR DE REEMPLAZOS
#  "Se me fue / lesionó un jugador: ¿quién juega parecido? ¿a quién
#   de otra posición podría adaptar?"
#
#  Idea:
#   1. El clúster (rol GMM) define QUÉ tipo de jugador busco.
#   2. La distancia euclídea en las 55 métricas estandarizadas (z-score)
#      ordena a los candidatos por parecido con el jugador a reemplazar.
#      (Se eligió este espacio porque es el que mejor reconoce a un mismo
#       jugador de una temporada a la siguiente; ver tabla en el README / ppt.)
#   3. "Adaptable" = candidato cuya posición nominal NO incluye la posición
#      principal del jugador a reemplazar (ej. un MF que juega como lateral).
#
#  Requiere haber corrido antes Clustering/clustering (genera asignacion_clusters.csv).
# =====================================================================
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
ruta_z = RAIZ_PROYECTO / 'datos_kaggle' / 'procesados' / 'matriz_master_pca.csv'
ruta_asig = RAIZ_PROYECTO / 'Clustering' / 'asignacion_clusters.csv'

META = ['league', 'season', 'team', 'player', 'id_jugador', 'nation_', 'pos_']

df_z = pd.read_csv(ruta_z)
asig = pd.read_csv(ruta_asig)

# Las dos tablas tienen que estar en el mismo orden (salen del mismo pipeline)
clave = ['id_jugador', 'season', 'team']
if not (df_z[clave].astype(str).values == asig[clave].astype(str).values).all():
    raise ValueError("matriz_master_pca.csv y asignacion_clusters.csv no están alineadas: "
                     "volvé a correr el pipeline completo.")

VARIABLES = [c for c in df_z.columns if c not in META]
Z = df_z[VARIABLES].to_numpy()
info = asig.copy()
nombres_rol = info.drop_duplicates('Cluster').set_index('Cluster')['Rol'].sort_index()


def _fila_jugador(id_jugador, temporada):
    idx = info.index[(info['id_jugador'] == id_jugador) & (info['season'] == temporada)]
    if len(idx) == 0:
        parecidos = info.loc[info['id_jugador'].str.contains(id_jugador.split(' (')[0], case=False, regex=False),
                             ['id_jugador', 'season', 'team']].drop_duplicates()
        raise KeyError(f"No encontré a '{id_jugador}' en la temporada {temporada} "
                       f"(¿tiene menos de 900 min?). Coincidencias:\n{parecidos.to_string(index=False)}")
    if len(idx) > 1:
        print(f"Aviso: {id_jugador} jugó en {len(idx)} equipos en {temporada}; "
              f"uso la fila de {info.loc[idx[0], 'team']}.")
    return idx[0]


def distancia_tipica_propia():
    """Distancia mediana entre un mismo jugador y él mismo la temporada siguiente.
    Sirve de vara: un candidato a menos de esta distancia se parece a tu jugador
    tanto como el propio jugador se parece a sí mismo de un año al otro."""
    temporadas = sorted(info['season'].unique())
    dists = []
    for a, b in zip(temporadas[:-1], temporadas[1:]):
        ia = info[info['season'] == a].drop_duplicates('id_jugador', keep=False)
        ib = info[info['season'] == b].drop_duplicates('id_jugador', keep=False)
        comunes = ia.reset_index().merge(ib.reset_index(), on='id_jugador')
        dists.append(np.linalg.norm(Z[comunes['index_x']] - Z[comunes['index_y']], axis=1))
    return float(np.median(np.concatenate(dists)))


DIST_PROPIA = distancia_tipica_propia()


def buscar_reemplazos(id_jugador, temporada, temporada_busqueda=None, n=10,
                      rol=None, prob_min=0.5, solo_adaptables=False, excluir_equipo=False):
    """
    id_jugador         : ej. 'Trent Alexander-Arnold (1998)'
    temporada          : temporada del perfil a reemplazar, ej. 2425
    temporada_busqueda : en qué temporada busco candidatos (por defecto la misma)
    rol                : número de clúster objetivo (por defecto, el del jugador)
    prob_min           : probabilidad mínima de pertenecer al rol objetivo
    solo_adaptables    : True = solo jugadores de otra posición nominal
    excluir_equipo     : True = sacar a los compañeros de equipo
    """
    i = _fila_jugador(id_jugador, temporada)
    jugador = info.loc[i]
    temporada_busqueda = temporada if temporada_busqueda is None else temporada_busqueda
    rol = int(jugador['Cluster']) if rol is None else rol
    pos_principal = str(jugador['pos_']).split(',')[0]

    cand = info[(info['season'] == temporada_busqueda) & (info['id_jugador'] != id_jugador)].copy()
    cand['Prob_Rol'] = cand[f'Prob_C{rol}']
    cand = cand[cand['Prob_Rol'] >= prob_min]
    if excluir_equipo:
        cand = cand[cand['team'] != jugador['team']]
    cand['Adaptable'] = ~cand['pos_'].astype(str).str.split(',').apply(lambda p: pos_principal in p)
    if solo_adaptables:
        cand = cand[cand['Adaptable']]

    cand['Distancia'] = np.linalg.norm(Z[cand.index] - Z[i], axis=1)
    cand['Dist_rel'] = cand['Distancia'] / DIST_PROPIA   # < 1: más parecido que él mismo año a año
    cand = cand.sort_values('Distancia').head(n)

    print(f"\n{'=' * 90}\n Reemplazos para {id_jugador} ({jugador['team']}, {temporada}, {jugador['pos_']})"
          f" — rol: {nombres_rol[rol]}\n Buscando en {temporada_busqueda}"
          f"{' | solo adaptables' if solo_adaptables else ''}{' | sin compañeros' if excluir_equipo else ''}"
          f" | prob. del rol ≥ {prob_min}\n{'=' * 90}")
    cols = ['id_jugador', 'team', 'league', 'pos_', 'Rol', 'Prob_Rol', 'Distancia', 'Dist_rel', 'Adaptable']
    print(cand[cols].round(2).to_string(index=False))
    return cand[cols]


def explicar_parecido(id_a, temporada_a, id_b, temporada_b, n=5):
    """Muestra en qué métricas se parecen más y en cuáles difieren más dos jugadores."""
    ia, ib = _fila_jugador(id_a, temporada_a), _fila_jugador(id_b, temporada_b)
    comp = pd.DataFrame({id_a: Z[ia], id_b: Z[ib]}, index=[v.replace('_p90', '') for v in VARIABLES])
    comp['Diferencia'] = comp[id_b] - comp[id_a]
    orden = comp['Diferencia'].abs().sort_values()
    print(f"\n--- {id_b} ({temporada_b}) vs {id_a} ({temporada_a}) — valores en z-score ---")
    print("Más parecidos en:")
    print(comp.loc[orden.index[:n]].round(2).to_string())
    print("Más distintos en:")
    print(comp.loc[orden.index[::-1][:n]].round(2).to_string())
    return comp


print(f"Distancia típica de un jugador consigo mismo de una temporada a la siguiente: {DIST_PROPIA:.2f}")

#%%
# ---------------------------------------------------------------------
# CASO 1: Trent se va del Liverpool al Real Madrid (verano 2025).
# ¿A quién traigo para jugar como él en 2025/26? Uso su perfil de 2024/25.
# ---------------------------------------------------------------------
TRENT = 'Trent Alexander-Arnold (1998)'
trent_todos = buscar_reemplazos(TRENT, 2425, n=10, excluir_equipo=True)
trent_adapt = buscar_reemplazos(TRENT, 2425, n=10, solo_adaptables=True, excluir_equipo=True)

#%%
# ---------------------------------------------------------------------
# CASO 2: Rodri se rompe los ligamentos (septiembre 2024) y se pierde 2024/25.
# Busco en 2024/25 jugadores parecidos a su versión 2023/24.
# ---------------------------------------------------------------------
RODRI = 'Rodri (1996)'
rodri_todos = buscar_reemplazos(RODRI, 2324, temporada_busqueda=2425, n=10, excluir_equipo=True)
rodri_adapt = buscar_reemplazos(RODRI, 2324, temporada_busqueda=2425, n=10, solo_adaptables=True,
                                excluir_equipo=True)
# %%
