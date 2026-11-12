from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.io import (
    SNDLIB_NAMESPACE,
    SndlibFormatError,
    read_sndlib,
    write_sndlib,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "check/data/sndlib"


class SndlibIoTests(unittest.TestCase):
    def test_reads_capacity_cost_and_metadata_variants(self) -> None:
        instance = read_sndlib(DATA / "atlanta.xml")
        self.assertEqual(instance.version, "1.0")
        self.assertEqual(instance.coordinates_type, "pixel")
        self.assertEqual(len(instance.nodes), 15)
        self.assertEqual(len(instance.links), 22)
        self.assertEqual(len(instance.demands), 210)
        self.assertIsNone(instance.routing_unit)

        link = instance.links[0]
        self.assertEqual((link.source, link.target), ("N1", "N6"))
        self.assertEqual(link.routing_cost, 67.8)
        preinstalled = link.preinstalled_module
        self.assertIsNotNone(preinstalled)
        assert preinstalled is not None
        self.assertEqual(preinstalled.capacity, 11000.0)
        self.assertEqual([module.capacity for module in link.additional_modules], [1000.0, 4000.0])

        abilene = read_sndlib(DATA / "abilene.xml")
        self.assertEqual(abilene.routing_unit, "MBITPERSEC")
        metadata = abilene.metadata
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.granularity, "6month")

    def test_reads_routing_restriction_variants(self) -> None:
        constrained = read_sndlib(DATA / "dfn-gwin.xml")
        self.assertEqual(constrained.demands[0].max_path_length, 2)

        path_routed = read_sndlib(DATA / "polska.xml")
        first_path = path_routed.demands[0].admissible_paths[0]
        self.assertEqual(first_path.id, "P_0")
        self.assertEqual(first_path.link_ids, ("Link_0_2", "Link_1_2"))

    def test_round_trip_preserves_every_downloaded_instance(self) -> None:
        paths = sorted(DATA.glob("*.xml"))
        self.assertEqual(len(paths), 26)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for source in paths:
                with self.subTest(instance=source.stem):
                    original = read_sndlib(source)
                    reread = read_sndlib(write_sndlib(original, output / source.name))
                    self.assertEqual(reread, original)

    def test_writer_is_deterministic_and_namespace_qualified(self) -> None:
        instance = read_sndlib(DATA / "abilene.xml")
        with tempfile.TemporaryDirectory() as directory:
            first = write_sndlib(instance, Path(directory) / "first.xml")
            second = write_sndlib(instance, Path(directory) / "second.xml")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertIn(f'xmlns="{SNDLIB_NAMESPACE}"'.encode(), first.read_bytes())

    def test_graph_projections_do_not_guess_link_semantics(self) -> None:
        instance = read_sndlib(DATA / "janos-us.xml")
        directed = instance.directed_link_graph()
        topology = instance.topology()
        self.assertEqual(directed.number_of_edges(), len(instance.links))
        self.assertEqual(topology.number_of_edges(), len(instance.links) // 2)
        self.assertEqual(topology["Seattle"]["SanFrancisco"]["multiplicity"], 2)
        self.assertEqual(
            topology["Seattle"]["SanFrancisco"]["orientations"],
            (("Seattle", "SanFrancisco"), ("SanFrancisco", "Seattle")),
        )

    def test_rejects_wrong_namespace(self) -> None:
        xml = b"""<?xml version="1.0"?>
<network version="1.0">
 <networkStructure><nodes coordinatesType="pixel"></nodes><links></links></networkStructure>
 <demands></demands>
</network>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.xml"
            path.write_bytes(xml)
            with self.assertRaisesRegex(SndlibFormatError, "namespace"):
                read_sndlib(path)

    def test_rejects_disconnected_admissible_path(self) -> None:
        text = (DATA / "polska.xml").read_text(encoding="ISO-8859-1")
        text = text.replace("<linkId>Link_0_2</linkId>", "<linkId>Link_3_4</linkId>", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.xml"
            path.write_text(text, encoding="ISO-8859-1")
            with self.assertRaisesRegex(SndlibFormatError, "not contiguous"):
                read_sndlib(path)


if __name__ == "__main__":
    unittest.main()
