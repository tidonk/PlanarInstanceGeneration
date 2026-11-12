from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import networkx as nx

from planar_instance_generation.core import coupling_pair_count, coupling_pair_rule
from planar_instance_generation.evaluation import run_power_experiment, run_snd_experiment
from tests.test_power_experiment import _triangulated_source
from tests.test_power_generation import _FeasibleGate as PowerGate
from tests.test_snd_generation import _source as snd_source, _FeasibleGate as SndGate
from planar_instance_generation.core import SndSolution
from planar_instance_generation.io import SndlibNode, SndlibLink, SndlibDemand, SndlibCapacityModule


class CouplingCountRuleTests(unittest.TestCase):
    def test_counts_round_up_and_respect_minimum_and_smaller_graph(self):
        for n, sqrt_count, log_count in [(1,1,1),(2,2,2),(3,3,3),(9,3,4),
                                        (16,4,4),(17,5,5),(64,8,6),(65,9,7)]:
            with self.subTest(n=n):
                small, large = nx.path_graph(n), nx.path_graph(100)
                self.assertEqual(coupling_pair_count(small, large, rule='sqrt'), sqrt_count)
                self.assertEqual(coupling_pair_count(large, small, rule='sqrt'), sqrt_count)
                self.assertEqual(coupling_pair_count(small, large), log_count)
                self.assertEqual(coupling_pair_count(small, large, rule='log2'), log_count)
        for rule in ('bad', None, []):
            with self.assertRaisesRegex(ValueError, 'log2.*sqrt'):
                coupling_pair_count(nx.path_graph(3), nx.path_graph(3), rule=rule)

    def test_both_experiments_propagate_rule_to_edges_and_metadata(self):
        source = snd_source('path')
        nodes = tuple(SndlibNode(str(i), i, i % 2) for i in range(9))
        links = tuple(SndlibLink(str(i), str(i), str(i+1),
                      additional_modules=(SndlibCapacityModule(10, 1),), routing_cost=1)
                      for i in range(8))
        source = replace(source, network=replace(source.network, nodes=nodes, links=links,
                         demands=()),
                         solution=SndSolution({str(i):10 for i in range(8)},
                                              {}))
        for runner, pool, gate in [(run_power_experiment, (_triangulated_source('nine',9),), PowerGate),
                                   (run_snd_experiment, (source,), SndGate)]:
            for rule, expected in [('log2',4),('sqrt',3)]:
                with self.subTest(domain=runner.__name__, rule=rule):
                    result = runner(pool, lambda *_: gate(), strategies=('random','high_degree'),
                                    requested_per_strategy=1, num_subinstances=(2,2),
                                    target_size=(18,18), size_range=(1,1), max_attempts_per_output=1,
                                    seed=1, coupling_count_rule=rule)
                    self.assertEqual(len(result.accepted),2)
                    self.assertFalse(result.failures)
                    for accepted in result.accepted:
                        candidate=accepted.candidate
                        self.assertEqual(candidate.stages[0].pair_count, expected)
                        self.assertEqual(candidate.statistics()['coupling_count_rule'],rule)
                        self.assertEqual(candidate.statistics()['coupling_rule'],coupling_pair_rule(rule))
                    self.assertTrue(all(r.candidate_statistics['coupling_count_rule']==rule for r in result.attempts))

    def test_face_capacity_keeps_the_requested_count_strict(self):
        from planar_instance_generation.evaluation import extract_power_pieces, merge_power_pieces
        from planar_instance_generation.evaluation.power_generation import PowerMergeSelectionError
        source=_triangulated_source('triangulated',9)
        pieces=extract_power_pieces((source,source),1.0,seed=1)
        for strategy in ('heuristic_outer_face','exact_outer_face'):
            with self.subTest(strategy=strategy):
                candidate=merge_power_pieces(pieces,PowerGate(),strategy=strategy,
                                             coupling_count_rule='sqrt')
                self.assertEqual(candidate.stages[0].pair_count,3)
                with self.assertRaises(PowerMergeSelectionError):
                    merge_power_pieces(pieces,PowerGate(),strategy=strategy,
                                       coupling_count_rule='log2')

    def test_invalid_rule_is_configuration_error_before_sampling(self):
        for runner in (run_power_experiment, run_snd_experiment):
            gate=Mock()
            with self.assertRaisesRegex(ValueError,'coupling count rule'):
                runner((), gate, strategies=('random',), requested_per_strategy=1,
                       num_subinstances=(2,2), target_size=(18,18), size_range=(1,1),
                       max_attempts_per_output=1, coupling_count_rule='typo')
            gate.assert_not_called()

    def test_generator_cli_defaults_and_explicit_rule(self):
        from check.scripts import generate_pglib, generate_sndlib
        for module in (generate_pglib, generate_sndlib):
            for argv, expected in [([], 'log2'), (['--coupling-count-rule','sqrt'], 'sqrt')]:
                with patch('sys.argv', [module.__name__, *argv]):
                    self.assertEqual(module.parse_args().coupling_count_rule,expected)
