import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check.scripts.collect_snd_flow_measurements import collect, METHODS


class SndFlowCollectionTests(unittest.TestCase):
    def inputs(self, root):
        (root / 'inputs').mkdir()
        for index, method in enumerate(METHODS):
            stem = f'case_{index}'
            source = root / 'inputs' / (stem + '.json.gz')
            with gzip.open(source, 'wt') as stream:
                json.dump({'stem': stem}, stream)
            result = dict(stem=stem, strategy=method, input_index=1, success=True,
                          primary_status=2, secondary_status=2,
                          input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          coupling={'mean': 2.0}, inherited={'mean': 4.0},
                          ratios={'mean': .5}, positive_coupling_edges=1)
            path = root / 'results' / stem / 'summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(result))

    def test_complete_results_retain_individual_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.inputs(root)
            report = collect(root, root / 'collected', expected_inputs=1)
            self.assertEqual(report['successful'], 5)
            results = json.loads((root / 'collected/results.json').read_text())['results']
            self.assertEqual(len(results), 5)
            self.assertTrue(all(row['ratios']['mean'] == .5 for row in results))

    def test_changed_input_and_unfinished_solve_are_rejected(self):
        for failure in ('input', 'status', 'ratio', 'missing'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.inputs(root)
                path = root / 'results/case_0/summary.json'
                row = json.loads(path.read_text())
                if failure == 'input':
                    (root / 'inputs/case_0.json.gz').write_bytes(b'changed')
                elif failure == 'status':
                    row['secondary_status'] = 9
                    path.write_text(json.dumps(row))
                elif failure == 'ratio':
                    row['ratios']['mean'] = 2
                    path.write_text(json.dumps(row))
                else:
                    path.unlink()
                with self.assertRaises(ValueError):
                    collect(root, root / 'collected', expected_inputs=1)
                self.assertFalse((root / 'collected').exists())


class SndFlowOptimizationTests(unittest.TestCase):
    def test_cost_optimum_and_opposing_commodity_flows(self):
        try:
            from check.scripts.run_snd_flow_measurements import solve
        except ImportError:
            self.skipTest('gurobipy is not installed')
        data = dict(nodes=['A', 'B', 'C', 'D'], baseline_objective=160,
                    edges=[dict(id='ab', source='A', target='B', cost=10, capacity=15, coupling=False),
                           dict(id='cd', source='C', target='D', cost=1, capacity=35, coupling=False),
                           dict(id='ac', source='A', target='C', cost=0, capacity=20, coupling=True),
                           dict(id='bd', source='B', target='D', cost=0, capacity=20, coupling=True)],
                    demands=[dict(id='forward', source='A', target='B', value=10),
                             dict(id='reverse', source='B', target='A', value=5),
                             dict(id='right', source='C', target='D', value=10)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            baseline = solve(data, path, baseline=True)
            result = solve(data, path)
            self.assertAlmostEqual(baseline['primary_objective'], 160, places=5)
            self.assertAlmostEqual(result['primary_objective'], 25, places=5)
            self.assertAlmostEqual(result['coupling']['mean'], 15, places=5)
            self.assertAlmostEqual(result['inherited']['mean'], 12.5, places=5)
            self.assertAlmostEqual(result['ratios']['mean'], 1.2, places=5)
            self.assertAlmostEqual(result['secondary_objective'], 55, places=5)


if __name__ == '__main__':
    unittest.main()
