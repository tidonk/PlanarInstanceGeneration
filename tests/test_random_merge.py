from __future__ import annotations

import unittest

import networkx as nx

from planar_instance_generation.core import merge_with_pairs, random_merge, random_merge_sequence


class RandomMergeTests(unittest.TestCase):
    def test_namespaces_colliding_node_ids_and_preserves_inputs(self) -> None:
        left = nx.path_graph([1, 2, 3])
        right = nx.path_graph([1, 2])
        left.nodes[1]["kind"] = "source"
        before_left = left.copy()
        before_right = right.copy()

        result = random_merge(left, right, 3, seed=7)

        self.assertEqual(set(result.initial_graph), {
            ("left", 1), ("left", 2), ("left", 3), ("right", 1), ("right", 2)
        })
        self.assertEqual(result.final_graph.number_of_edges(), 3 + 3)
        self.assertEqual(result.final_graph.nodes[("left", 1)]["kind"], "source")
        self.assertTrue(nx.utils.graphs_equal(left, before_left))
        self.assertTrue(nx.utils.graphs_equal(right, before_right))

    def test_seed_is_reproducible_and_edges_are_unique(self) -> None:
        left = nx.path_graph(6)
        right = nx.cycle_graph(5)
        first = random_merge(left, right, 12, seed=123)
        second = random_merge(left, right, 12, seed=123)
        self.assertEqual(first.added_edges, second.added_edges)
        self.assertEqual(len(set(first.added_edges)), 12)

    def test_same_object_means_two_copies(self) -> None:
        source = nx.path_graph(3)
        result = random_merge(source, source, 1, seed=1)
        self.assertEqual(result.initial_graph.number_of_nodes(), 6)
        self.assertTrue(all(edge[0][0] == "left" and edge[1][0] == "right" for edge in result.added_edges))

    def test_sequence_preserves_original_provenance_across_stages(self) -> None:
        graphs = [nx.path_graph(3), nx.cycle_graph(4), nx.star_graph(3)]
        sequence = random_merge_sequence(graphs, [2, 3], seed=9)
        self.assertEqual(len(sequence.stages), 2)
        self.assertEqual(sequence.source_ids, ("G_1", "G_2", "G_3"))
        final_sources = {data["source_graph"] for _, data in sequence.final_graph.nodes(data=True)}
        self.assertEqual(final_sources, {"G_1", "G_2", "G_3"})
        internal_edge_sources = {
            data["source_graph"]
            for _, _, data in sequence.final_graph.edges(data=True)
            if not data.get("merge_edge")
        }
        self.assertEqual(internal_edge_sources, {"G_1", "G_2", "G_3"})
        self.assertEqual(
            {data["merge_stage"] for _, _, data in sequence.final_graph.edges(data=True) if data.get("merge_edge")},
            {1, 2},
        )

    def test_sequence_validates_lengths_and_source_ids(self) -> None:
        graphs = [nx.path_graph(2), nx.path_graph(2), nx.path_graph(2)]
        with self.assertRaises(ValueError):
            random_merge_sequence(graphs, [1])
        with self.assertRaises(ValueError):
            random_merge_sequence(graphs, 1, source_ids=["same", "same", "other"])

    def test_step_reconstruction(self) -> None:
        result = random_merge(nx.path_graph(3), nx.path_graph(4), 3, seed=2)
        self.assertEqual(result.graph_at_step(0).number_of_edges(), result.initial_graph.number_of_edges())
        self.assertTrue(nx.utils.graphs_equal(result.graph_at_step(3), result.final_graph))
        steps = list(result.iter_steps())
        self.assertEqual([step for step, _, _ in steps], [1, 2, 3])
        self.assertEqual(steps[-1][1].edges, result.final_graph.edges)

    def test_reports_separate_nonnegative_timings(self) -> None:
        result = random_merge(nx.path_graph(10), nx.path_graph(10), 5, seed=0)
        self.assertGreaterEqual(result.selection_seconds, 0.0)
        self.assertGreaterEqual(result.addition_seconds, 0.0)
        self.assertEqual(
            result.total_algorithm_seconds,
            result.selection_seconds + result.addition_seconds,
        )

    def test_merges_preselected_endpoint_disjoint_pairs(self) -> None:
        result = merge_with_pairs(
            nx.path_graph(4),
            nx.path_graph(range(10, 14)),
            [(0, 10), (3, 13)],
            selection_seconds=0.25,
            seed=5,
            stage_index=2,
        )
        self.assertEqual(result.selection_seconds, 0.25)
        self.assertEqual(
            result.added_edges,
            ((('left', 0), ('right', 10)), (('left', 3), ('right', 13))),
        )
        self.assertTrue(all(result.final_graph.edges[edge]["merge_edge"] for edge in result.added_edges))

        with self.assertRaisesRegex(ValueError, "reuse"):
            merge_with_pairs(
                nx.path_graph(3),
                nx.path_graph(3),
                [(0, 0), (0, 1)],
                selection_seconds=0.0,
            )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            random_merge(nx.Graph(), nx.path_graph(2), 1)
        with self.assertRaises(TypeError):
            random_merge(nx.DiGraph([(1, 2)]), nx.path_graph(2), 1)
        with self.assertRaises(TypeError):
            random_merge(nx.MultiGraph([(1, 2)]), nx.path_graph(2), 1)
        with self.assertRaises(ValueError):
            random_merge(nx.path_graph(2), nx.path_graph(2), 0)
        with self.assertRaises(ValueError):
            random_merge(nx.path_graph(2), nx.path_graph(2), 5)


if __name__ == "__main__":
    unittest.main()
