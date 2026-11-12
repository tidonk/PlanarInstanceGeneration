"""Voltage subgraph certificates strengthen bounds without changing the graph."""

import unittest
from unittest import mock

import networkx as nx

from planar_instance_generation.core.planarization import maximum_planar_subgraph
from planar_instance_generation.evaluation.voltage_certificate import lift_subgraph_certificate


class VoltageCertificateTests(unittest.TestCase):
    def certificate(self):
        separated = nx.disjoint_union(nx.complete_graph(5), nx.cycle_graph(4))
        original = separated.copy()
        original.add_edge(0, 5, voltage_crossing=True)
        planar = separated.copy()
        planar.remove_edge(0, 1)
        return original, separated, planar

    def test_lift_keeps_bound_and_restores_components_by_bridges(self):
        original, separated, planar = self.certificate()
        piece = lift_subgraph_certificate(original, separated, planar, 1)
        self.assertEqual(piece.lower_bound, 1)
        self.assertEqual(nx.number_connected_components(piece.planar_subgraph), 1)
        self.assertTrue(nx.check_planarity(piece.planar_subgraph)[0])
        self.assertTrue(piece.planar_subgraph[0][5]["voltage_crossing"])
        self.assertEqual(planar.number_of_edges(), separated.number_of_edges() - 1)

    def test_matching_positive_bounds_skip_the_optimizer(self):
        original, separated, planar = self.certificate()
        piece = lift_subgraph_certificate(original, separated, planar, 1)
        with mock.patch("planar_instance_generation.core.ogdf.build_identity", return_value={}):
            result = maximum_planar_subgraph(original, certified_pieces=(piece,),
                                             time_limit_seconds=1)
        self.assertEqual((result.omitted_lower_bound, result.omitted_upper_bound), (1, 1))
        self.assertEqual(result.termination_status, "OPTIMAL")
        self.assertIsNone(result.optimizer)

    def test_invalid_certificates_are_rejected(self):
        original, separated, planar = self.certificate()
        for invalid, bound in ((separated, 1), (planar, 2), (planar, -1)):
            with self.assertRaises(ValueError):
                lift_subgraph_certificate(original, separated, invalid, bound)
        planar.add_edge(0, 5)
        with self.assertRaises(ValueError):
            lift_subgraph_certificate(original, separated, planar, 1)

    def test_original_disconnected_components_are_not_joined(self):
        _, separated, planar = self.certificate()
        piece = lift_subgraph_certificate(separated, separated, planar, 1)
        self.assertEqual(nx.number_connected_components(piece.planar_subgraph), 2)
