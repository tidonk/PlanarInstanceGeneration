from __future__ import annotations

from dataclasses import replace
import unittest

from planar_instance_generation.core import (
    SndCommodity,
    SndInstance,
    SndSolution,
    build_snd_instance,
)
from planar_instance_generation.core.snd_isomorphism import (
    snd_isomorphic,
    sndlib_isomorphic,
)
from planar_instance_generation.io import (
    SndlibCapacityModule,
    SndlibDemand,
    SndlibLink,
    SndlibNetwork,
    SndlibNode,
)


def _network(*, reverse_link: bool = False, module_cost: float = 3.0) -> SndlibNetwork:
    return SndlibNetwork(
        version="1.0",
        coordinates_type="pixel",
        nodes=(SndlibNode("a", 0, 0), SndlibNode("b", 2, 1)),
        links=(
            SndlibLink(
                "edge",
                "b" if reverse_link else "a",
                "a" if reverse_link else "b",
                additional_modules=(SndlibCapacityModule(5, module_cost),),
                routing_cost=0.5,
                setup_cost=1.0,
            ),
        ),
        demands=(SndlibDemand("demand", "a", "b", 2),),
    )


class SndIsomorphismTests(unittest.TestCase):
    def test_xml_link_reference_orientation_is_ignored(self) -> None:
        self.assertTrue(sndlib_isomorphic(_network(), _network(reverse_link=True)))

        left = build_snd_instance(
            _network(),
            SndSolution({"edge": 5}, {("demand", "edge"): 2}),
            name="left",
        )
        right = build_snd_instance(
            _network(reverse_link=True),
            SndSolution({"edge": 5}, {("demand", "edge"): -2}),
            name="right",
        )
        self.assertTrue(snd_isomorphic(left, right))

    def test_serialized_model_attributes_are_preserved(self) -> None:
        changed_cost = _network(module_cost=4.0)
        self.assertFalse(sndlib_isomorphic(_network(), changed_cost))
        reversed_demand = replace(
            _network(),
            demands=(SndlibDemand("other", "b", "a", 2),),
        )
        self.assertFalse(sndlib_isomorphic(_network(), reversed_demand))

    def test_additional_module_row_order_is_ignored(self) -> None:
        first = SndlibCapacityModule(5, 3)
        second = SndlibCapacityModule(8, 4)
        left = replace(
            _network(),
            links=(
                replace(
                    _network().links[0],
                    additional_modules=(first, second),
                ),
            ),
        )
        right = replace(
            left,
            links=(
                replace(
                    left.links[0],
                    additional_modules=(second, first),
                ),
            ),
        )
        self.assertTrue(sndlib_isomorphic(left, right))

    def test_auxiliary_reference_state_does_not_change_release_identity(self) -> None:
        base = build_snd_instance(
            _network(),
            SndSolution({"edge": 5}, {("demand", "edge"): 2}),
            name="base",
        )
        with_inert = SndInstance(
            name="with_inert",
            version=base.version,
            coordinates_type=base.coordinates_type,
            nodes=base.nodes,
            links=base.links,
            commodities=(
                *base.commodities,
                SndCommodity("inert", {}, "provenance", None),
            ),
            reference_solution=SndSolution(
                {"edge": 10},
                base.reference_solution.flows,
            ),
        )
        self.assertTrue(snd_isomorphic(base, with_inert))


if __name__ == "__main__":
    unittest.main()
