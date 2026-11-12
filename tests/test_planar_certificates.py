import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from planar_instance_generation.core.planarization import CertifiedPlanarPiece, maximum_planar_subgraph
from planar_instance_generation.evaluation.planar_certificates import generation_certificates, attach_generation_certificates
from planar_instance_generation.evaluation.planarization_batch import PlanarizationTask, solve_planarization_batch


class PlanarCertificateTests(unittest.TestCase):
    def test_five_piece_generation_preserves_all_stage_certificates(self):
        from tests.test_power_generation import _source, _FeasibleGate
        from planar_instance_generation.evaluation import extract_power_pieces, merge_power_pieces
        pieces = extract_power_pieces(tuple(_source(f"source_{i}") for i in range(5)), 1.0, seed=2)
        candidate = merge_power_pieces(pieces, _FeasibleGate(), strategy="heuristic_outer_face", seed=3)
        graph = candidate.case.topology(active_only=True)
        records = json.loads(json.dumps(generation_certificates(candidate)))
        self.assertEqual(len(records), 8)
        for record in records:
            induced = graph.subgraph(record["nodes"])
            self.assertEqual({frozenset(e) for e in induced.edges()}, {frozenset(e) for e in record["edges"]})
            self.assertTrue(nx.check_planarity(nx.Graph(record["planar_edges"]))[0])

    def test_snd_namespace_remapping_preserves_certificates(self):
        from tests.test_snd_generation import _source, _FeasibleGate
        from planar_instance_generation.evaluation.snd_generation import extract_snd_pieces, merge_snd_pieces
        pieces = extract_snd_pieces(tuple(_source(str(i)) for i in range(5)), 1.0, seed=2)
        candidate = merge_snd_pieces(pieces, _FeasibleGate(), strategy="heuristic_outer_face", seed=3, source_stratum_size=5)
        graph = candidate.instance.topology()
        records = json.loads(json.dumps(generation_certificates(candidate)))
        self.assertEqual(len(records), 8)
        for record in records:
            self.assertEqual({frozenset(e) for e in graph.subgraph(record["nodes"]).edges()},
                             {frozenset(e) for e in record["edges"]})

    def test_overlapping_bounds_are_not_added_and_start_proves_optimality(self):
        graph = nx.complete_graph(5)
        planar = graph.copy()
        planar.remove_edge(0, 1)
        certificate = CertifiedPlanarPiece(graph, planar, 1)
        with patch("planar_instance_generation.core.ogdf.build_identity", return_value={}):
            result = maximum_planar_subgraph(graph, time_limit_seconds=1,
                                            prior_certificates=(certificate, certificate))
        self.assertEqual((result.omitted_lower_bound, result.omitted_upper_bound), (1, 1))
        self.assertTrue(result.exact)

    def test_equal_global_bounds_retain_distinct_subgraph_bounds(self):
        graph = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
        graph.add_edge(0, 5)
        kept = graph.copy()
        kept.remove_edges_from([(0, 1), (5, 6)])
        common = dict(nodes=list(graph), edges=list(graph.edges),
                      planar_edges=list(kept.edges), lower_bound=1)
        metadata = {
            'source_mappings': [{'source_artifact_stem': 'source',
                                 'node_ids': list(zip(graph, graph))}],
            'planarization_certificates': [
                dict(common, bound_rows=[(list(graph.subgraph(range(5)).edges), 1)]),
                dict(common, bound_rows=[(list(graph.subgraph(range(5, 10)).edges), 1)]),
            ],
        }
        task = PlanarizationTask(graph, 0, Path('unused'), 'final')
        result, = attach_generation_certificates([task],
            {'final': [('piece', dict(zip(graph, graph)))]}, [(metadata, ['piece'])])
        from planar_instance_generation.core.ogdf import bound_rows
        self.assertEqual(len(result.prior_certificates), 2)
        self.assertEqual(bound_rows(graph, prior=result.prior_certificates)[2], 2)

    def test_transfer_accounts_for_removed_edges(self):
        original = nx.complete_graph(5)
        planar = original.copy()
        planar.remove_edge(0, 1)
        metadata = {"source_mappings": [{"source_artifact_stem": "source", "node_ids": list(zip(range(5), range(5)))}],
                    "planarization_certificates": [{"nodes": list(original), "edges": list(original.edges()),
                    "planar_edges": list(planar.edges()), "lower_bound": 1}]}
        tasks = [PlanarizationTask(original, 1, Path("unused"), "same"),
                 PlanarizationTask(planar, 1, Path("unused"), "changed")]
        dependencies = {name: [("piece", {n: n for n in original})] for name in ("same", "changed")}
        prepared = attach_generation_certificates(tasks, dependencies, [(metadata, ["piece"])])
        self.assertEqual([t.prior_certificates[0].lower_bound for t in prepared], [1, 0])
        self.assertTrue(all(nx.check_planarity(t.prior_certificates[0].planar_subgraph)[0] for t in prepared))

    def test_changed_budget_does_not_import_previous_solutions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = PlanarizationTask(nx.cycle_graph(5), 1, root / "solve.log", "cycle")
            kwargs = dict( solver_threads=1, workers=1, checkpoint_directory=root)
            solve_planarization_batch([task], time_limit_seconds=1, **kwargs)
            with patch("planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph", wraps=maximum_planar_subgraph) as solve:
                solve_planarization_batch([task], time_limit_seconds=2, **kwargs)
                self.assertEqual(solve.call_args.kwargs["prior_certificates"], ())
            self.assertEqual(len(list(root.glob("cycle.previous.*.json"))), 0)

    def test_repeated_identical_piece_tasks_remain_distinct_vertices(self):
        graph = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
        graph.add_edge(0, 3)
        metadata = {"source_mappings": [
            {"source_artifact_stem": str(i), "node_ids": [(n, n + 3*i) for n in range(3)]}
            for i in range(2)], "planarization_certificates": [{"nodes": list(graph),
            "edges": list(graph.edges()), "planar_edges": list(graph.edges()), "lower_bound": 0}]}
        piece = PlanarizationTask(nx.cycle_graph(3), 1, Path("unused"), "piece")
        final = PlanarizationTask(graph, 1, Path("unused"), "final")
        dependencies = {"final": [("piece", {n: n + 3*i for n in range(3)}) for i in range(2)]}
        prepared = attach_generation_certificates([piece, final], dependencies, [(metadata, ["piece", "piece"])])
        self.assertEqual(len(prepared[0].prior_certificates[0].graph), 3)
        self.assertEqual(len(prepared[1].prior_certificates[0].graph), 6)
        self.assertEqual(prepared[1].prior_certificates[0].planar_subgraph.number_of_edges(), 7)

    def test_cluster_budget_and_release_dependencies(self):
        from check.eval.make_reuse_dag import build_dag
        dag = build_dag("new_run", Path("/tmp/settings.env"), instances=2)
        self.assertEqual(sum(line.startswith("JOB gen_") for line in dag.splitlines()), 4)
        self.assertEqual(sum(line.startswith("JOB skew_") for line in dag.splitlines()), 20)
        self.assertEqual(dag.count('time_limit="3600"'), 20)
        self.assertIn("PARENT gen_snd_i002 CHILD skew_snd_i002_random", dag)
        self.assertIn("PARENT skew_snd_i002_exact_outer_face CHILD data_snd", dag)
        self.assertIn("JOB skew_snd_i002_random check/eval/snd_cohort_skewness.sub", dag)
        self.assertIn("PARENT skew_pglib_i002_exact_outer_face CHILD data_pglib", dag)
        self.assertNotIn("prepare.sub", dag)
        self.assertNotIn("JOB report_", dag)
        self.assertEqual(dag.count('data_only="1"'), 2)

    def test_application_release_is_local_to_generated_set(self):
        from check.eval.make_reuse_dag import build_dag
        dag = build_dag("new_run", Path("/tmp/settings.env"), instances=2)
        lines = dag.splitlines()
        parents = {}
        for line in lines:
            if line.startswith("PARENT "):
                _, parent, _, child = line.split()
                parents.setdefault(child, set()).add(parent)
        for dataset in ("pglib", "snd"):
            for index in (1, 2):
                app = f"app_{dataset}_i{index:03d}"
                self.assertEqual(parents[app], {f"gen_{dataset}_i{index:03d}"})
                self.assertIn(
                    f"check/eval/prepare_application.sh {dataset} new_run {index}", dag
                )
                self.assertIn(
                    f'jobs_file="check/eval/application_jobs/new_run/{dataset}_i{index:03d}.jobs"',
                    dag,
                )
            self.assertEqual(parents[f"summary_{dataset}"], {
                f"data_{dataset}", f"app_{dataset}_i001", f"app_{dataset}_i002",
            })
            self.assertTrue(all(parent.startswith("skew_")
                                for parent in parents[f"data_{dataset}"]))
            self.assertNotIn(f"JOB app_{dataset} ", dag)
        self.assertEqual(sum(line.startswith("JOB app_") for line in lines), 4)
        self.assertEqual(dag.count('job_priority="0"'), 4)
        self.assertEqual(dag.count('job_priority="100"'), 20)

    def test_source_screening_precedes_every_pglib_generation_job(self):
        from check.eval.make_reuse_dag import build_dag
        dag = build_dag("new_run", Path("/tmp/settings.env"), instances=2, prepare_sources=True, source_files=(Path("a.m"), Path("b.m")))
        self.assertIn("JOB sources_pglib check/eval/pglib_source_pool.sub", dag)
        self.assertEqual(dag.count("check/eval/pglib_sources.sub"), 2)
        self.assertIn("PARENT source_pglib_001 CHILD sources_pglib", dag)
        self.assertIn("PARENT source_pglib_002 CHILD sources_pglib", dag)
        self.assertEqual(dag.count('job_priority="100"'), 20)
        self.assertIn("PRIORITY skew_snd_i002_random 100", dag)
        for i in (1, 2):
            self.assertIn(f"PARENT sources_pglib CHILD gen_pglib_i{i:03d}", dag)
        self.assertNotIn("PARENT sources_pglib CHILD gen_snd", dag)
