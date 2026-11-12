"""Check input identity and publication exports without running experiments."""
import csv
import hashlib
import json
from unittest.mock import patch
import tempfile
from pathlib import Path
import unittest

from check.scripts.fetch_inputs import verify
from check.scripts.export_paper_skewness_tables import _read, STRATEGIES
from check.scripts.export_paper_skewness_cdf import export_skewness_pgf


class ReproductionTests(unittest.TestCase):
    def write_paired(self, path, extra=None):
        fields = ['output_index','strategy','skewness_inflation_lower_bound',
                  'skewness_inflation_upper_bound','skewness_normalized_inflation_upper_bound']
        records = [dict(zip(fields,[i,m,0,0,0])) for i in range(1,101) for m in STRATEGIES]
        if extra: records.append(dict(records[0],**extra))
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
        return records

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input';p.write_bytes(b'original')
            expected=hashlib.sha256(p.read_bytes()).hexdigest();verify(p,expected)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(p,expected)

    def test_complete_and_unique_publication_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'paired.csv';self.write_paired(p)
            self.assertEqual(len(_read(p)),100)
            self.write_paired(p,{'skewness_inflation_upper_bound':0})
            with self.assertRaises(ValueError):_read(p)

    def test_nonfinite_publication_bound_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'paired.csv';self.write_paired(p)
            s=p.read_text().replace('1,random,0,0,0','1,random,nan,0,0',1);p.write_text(s)
            with self.assertRaises(ValueError):_read(p)

    def test_source_tables_check_input_hashes(self):
        from check.scripts import export_paper_source_skewness as source
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'paper/resources/evaluation';out.mkdir(parents=True)
            data=root/'check/data/pglib';data.mkdir(parents=True)
            (data/'pglib_opf_test.m').write_text('test input')
            digest=hashlib.sha256((data/'pglib_opf_test.m').read_bytes()).hexdigest()
            original=root/'original';original.mkdir();voltage=root/'voltage';voltage.mkdir()
            record=dict(source='pglib_opf_test',source_sha256=digest,node_count=5,edge_count=8,
                        skewness_lower_bound=0,skewness_upper_bound=1,voltage_separated=False)
            (original/'pglib_opf_test.json').write_text(json.dumps(record))
            (voltage/'pglib_opf_test.json').write_text(json.dumps(record | {'voltage_separated':True}))
            snd=root/'snd.csv'
            rows=[dict(group='Original sources',instance=f'source{i}',node_count=5,edge_count=8,
                       skewness_lower_bound=0,skewness_upper_bound=1) for i in range(16)]
            with snd.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with patch.object(source,'REPOSITORY_ROOT',root):
                source.export(root/'paper',original,voltage,snd,['pglib_opf_test'])
                self.assertEqual(len(list(out.glob('*.tex'))),2)
                (data/'pglib_opf_test.m').write_text('changed input')
                with self.assertRaises(ValueError):
                    source.export(root/'paper',original,voltage,snd,['pglib_opf_test'])

    def test_cdf_exports_only_the_two_manuscript_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'resources/evaluation';out.mkdir(parents=True)
            p=root/'paired.csv';self.write_paired(p)
            export_skewness_pgf(out,{'pglib':p,'snd':p})
            self.assertEqual({p.name for p in out.iterdir()},
                {'plot-data','fig-eval-skewness-pglib.tikz','fig-eval-skewness-snd.tikz'})
            for data in (out/'plot-data').glob('*.dat'):
                self.assertEqual(data.read_text().splitlines()[1],'0 1')

if __name__ == '__main__':unittest.main()
