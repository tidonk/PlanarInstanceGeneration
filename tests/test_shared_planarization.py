from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import networkx as nx
from planar_instance_generation.core.ogdf import direct_solution

from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    maximum_planar_subgraph,
)
from planar_instance_generation.evaluation.planarization_batch import (
    PlanarizationTask,
    solve_planarization_batch,
    solve_partitioned_batch,
    source_mappings,
)


class SharedPlanarizationTests(unittest.TestCase):
    def test_nonplanar_coupling_is_checked_and_input_is_preserved(self):
        graph = nx.complete_bipartite_graph(3, 3)
        graph.edges[0, 3]["capacity"] = 7
        pieces = tuple(
            CertifiedPlanarPiece(graph.subgraph(nodes).copy(), nx.empty_graph(nodes), 0)
            for nodes in ((0, 1, 2), (3, 4, 5))
        )
        candidate = direct_solution(graph, pieces)
        self.assertTrue(nx.check_planarity(candidate)[0])
        self.assertEqual(candidate.number_of_edges(), 5)
        self.assertEqual(graph.number_of_edges(), 9)
        self.assertEqual(candidate.edges[0, 3]["capacity"], 7)
        self.assertEqual(
            set(candidate.edges()), set(direct_solution(graph, pieces).edges())
        )

    def test_partition_and_planar_certificate_are_validated(self):
        graph = nx.complete_graph(5)
        with self.assertRaisesRegex(ValueError, "must be planar"):
            direct_solution(graph, (CertifiedPlanarPiece(graph, graph, 0),))
        planar = graph.copy()
        planar.remove_edge(0, 1)
        piece = CertifiedPlanarPiece(graph, planar, 1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            direct_solution(graph, (piece, piece))
        with self.assertRaisesRegex(ValueError, "lower bound"):
            direct_solution(graph, (replace(piece, lower_bound=2),))

    def test_piece_solution_and_certified_bounds_reach_solver(self):
        graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
        graph.add_edge(0, 5)
        pieces = []
        for nodes, omitted in ((range(5), (0, 1)), (range(5, 10), (5, 6))):
            internal = graph.subgraph(nodes).copy()
            planar = internal.copy()
            planar.remove_edge(*omitted)
            pieces.append(CertifiedPlanarPiece(internal, planar, 1))
        result = maximum_planar_subgraph(
            graph,
            certified_pieces=tuple(pieces),
            time_limit_seconds=2,
        )
        self.assertEqual(
            (result.omitted_lower_bound, result.omitted_upper_bound), (2, 2)
        )
        self.assertEqual(result.shared_piece_lower_bound, 2)
        self.assertEqual(result.shared_piece_count, 2)
        self.assertLessEqual(
            result.omitted_upper_bound, result.greedy_start_upper_bound
        )
        self.assertLessEqual(result.omitted_upper_bound, result.piece_start_upper_bound)
        self.assertTrue(result.planar_subgraph.has_edge(0, 5))

    def test_five_methods_receive_identical_piece_certificates_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = nx.path_graph(3)
            graph = nx.disjoint_union(source, source)
            graph.add_edge(2, 3)
            tasks = [PlanarizationTask(source, 7, root / "source.log", "source")]
            dependencies = {}
            for strategy in (
                "random",
                "high_degree",
                "kamada_kawai",
                "heuristic_outer_face",
                "exact_outer_face",
            ):
                tasks.append(
                    PlanarizationTask(
                        graph, len(tasks), root / f"{strategy}.log", strategy
                    )
                )
                dependencies[strategy] = [
                    ("source", {i: i for i in range(3)}),
                    ("source", {i: i + 3 for i in range(3)}),
                ]
            results = solve_partitioned_batch(
                tasks,
                dependencies,
                time_limit_seconds=1,
                solver_threads=1,
                workers=1,
                checkpoint_directory=root / "checkpoints",
            )
            finals = results[1:]
            self.assertEqual(len({result.seed for result in finals}), 1)
            self.assertTrue(all(result.shared_piece_count == 2 for result in finals))
            self.assertTrue(all(result.omitted_upper_bound == 0 for result in finals))
            with mock.patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                side_effect=AssertionError("failed to reuse certificate"),
            ):
                cached = solve_partitioned_batch(
                    tasks,
                    dependencies,
                    time_limit_seconds=1,
                    solver_threads=1,
                    workers=1,
                    checkpoint_directory=root / "checkpoints",
                )
            self.assertEqual(
                [r.statistics() for r in results], [r.statistics() for r in cached]
            )

    def test_changed_piece_certificate_invalidates_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = nx.cycle_graph(3)
            path = graph.copy()
            path.remove_edge(0, 1)
            task = PlanarizationTask(
                graph,
                0,
                root / "final.log",
                "final",
                (CertifiedPlanarPiece(graph, path, 0),),
            )
            kwargs = dict(
                time_limit_seconds=1,
                solver_threads=1,
                workers=1,
                checkpoint_directory=root,
            )
            solve_planarization_batch((task,), **kwargs)
            changed = replace(
                task, certified_pieces=(CertifiedPlanarPiece(graph, graph, 0),)
            )
            with mock.patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                wraps=maximum_planar_subgraph,
            ) as solve:
                solve_planarization_batch((changed,), **kwargs)
            self.assertEqual(solve.call_count, 1)

            reordered = nx.Graph()
            reordered.add_nodes_from(graph)
            reordered.add_edges_from(reversed(list(graph.edges())))
            with mock.patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                wraps=maximum_planar_subgraph,
            ) as solve:
                solve_planarization_batch(
                    (replace(changed, graph=reordered),), **kwargs
                )
            self.assertEqual(solve.call_count, 1)

    def test_domain_mappings_preserve_node_types(self):
        self.assertEqual(
            source_mappings(
                {
                    "source_bus_mappings": [
                        {
                            "source_artifact_stem": "piece",
                            "buses": [{"source_bus": 1, "composite_bus": 9}],
                        }
                    ]
                }
            ),
            (("piece", {1: 9}),),
        )
        self.assertEqual(
            source_mappings(
                {
                    "source_mappings": [
                        {"source_artifact_stem": "piece", "node_ids": {"a": "copy_a"}}
                    ]
                }
            ),
            (("piece", {"a": "copy_a"}),),
        )


if __name__ == "__main__":
    unittest.main()
