import unittest

import pandas as pd

from Clustering.cluster_profiles import build_cluster_profiles, build_cluster_summary


class ClusterProfileTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "league": ["ENG", "ENG", "ESP", "ESP"],
                "season": [2425, 2425, 2425, 2425],
                "team": ["A", "B", "C", "D"],
                "id_jugador": ["Uno (1990)", "Uno (1990)", "Tres (1992)", "Cuatro (1993)"],
                "metric_attack": [2.0, 4.0, -1.0, -3.0],
                "metric_defense": [0.0, 2.0, -3.0, -1.0],
            }
        )
        self.assignments = pd.DataFrame(
            {
                "league": ["ENG", "ENG", "ESP", "ESP"],
                "season": [2425, 2425, 2425, 2425],
                "team": ["A", "B", "C", "D"],
                "id_jugador": ["Uno (1990)", "Uno (1990)", "Tres (1992)", "Cuatro (1993)"],
                "Cluster": [0, 0, 1, 1],
                "Rol": ["Atacante", "Atacante", "Defensor", "Defensor"],
            }
        )

    def test_build_profiles_reports_means_and_player_counts(self):
        profiles = build_cluster_profiles(
            self.data, self.assignments, ["metric_attack", "metric_defense"]
        ).set_index("Cluster")

        self.assertEqual(profiles.loc[0, "N_registros"], 2)
        self.assertEqual(profiles.loc[0, "N_jugadores_unicos"], 1)
        self.assertEqual(profiles.loc[0, "Rol"], "Atacante")
        self.assertAlmostEqual(profiles.loc[0, "metric_attack"], 3.0)
        self.assertAlmostEqual(profiles.loc[0, "metric_defense"], 1.0)
        self.assertAlmostEqual(profiles.loc[1, "metric_attack"], -2.0)

    def test_build_summary_lists_highest_and_lowest_metrics(self):
        profiles = build_cluster_profiles(
            self.data, self.assignments, ["metric_attack", "metric_defense"]
        )
        summary = build_cluster_summary(
            profiles, ["metric_attack", "metric_defense"], top_n=1, bottom_n=2
        ).set_index("Cluster")

        self.assertIn("metric_attack", summary.loc[0, "metricas_mas_altas"])
        self.assertIn("metric_defense", summary.loc[1, "metricas_mas_bajas"])
        self.assertEqual(summary.loc[1, "N_registros"], 2)

    def test_profiles_align_by_row_not_dataframe_index(self):
        data = self.data.set_axis([10, 11, 12, 13])
        assignments = self.assignments.set_axis([7, 8, 9, 10])

        profiles = build_cluster_profiles(
            data, assignments, ["metric_attack", "metric_defense"]
        ).set_index("Cluster")

        self.assertAlmostEqual(profiles.loc[0, "metric_attack"], 3.0)
        self.assertAlmostEqual(profiles.loc[1, "metric_attack"], -2.0)

    def test_rejects_misaligned_rows(self):
        misaligned = self.assignments.iloc[::-1].reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "orden"):
            build_cluster_profiles(
                self.data, misaligned, ["metric_attack", "metric_defense"]
            )


if __name__ == "__main__":
    unittest.main()
