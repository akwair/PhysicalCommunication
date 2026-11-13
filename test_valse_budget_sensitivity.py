"""Independent artifact checks for the selective VALSE iteration-budget audit."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parent


class ValseBudgetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'results/valse_budget_sensitivity.json').read_text())
        cls.old = []
        for i in (1,2,3):
            with np.load(ROOT/f'results/valse_check{i}/frame_metrics.npz') as data:
                cls.old.append({k:data[k].copy() for k in data.files})

    def test_exact_selection_and_original_preservation(self):
        expected = set()
        for campaign,data in enumerate(self.old):
            for key,val in data.items():
                if key.endswith('_valse_converged'):
                    for index in np.flatnonzero(~val):
                        expected.add((campaign,key.removesuffix('_valse_converged'),int(index)))
        actual = {(c['campaign'],f"{c['profile']}_{c['snr_db']}",c['index'])
                  for c in self.report['cases']}
        self.assertEqual(expected,actual)
        self.assertEqual(len(self.report['cases']),len(actual))
        self.assertEqual(len(actual),12)
        for path,sha in self.report['original_artifact_hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha)

    def test_independent_aggregate_by_additive_replacement(self):
        total = 0
        for cell in self.report['per_cell']:
            prefix = f"{cell['profile']}_{cell['snr_db']}"
            errors = np.concatenate([x[prefix+'_valse_equation'][0] for x in self.old])
            energies = np.concatenate([x[prefix+'_valse_equation'][1] for x in self.old])
            noise = np.concatenate([x[prefix+'_valse_noise_var'] for x in self.old])
            cases = [c for c in self.report['cases']
                     if c['profile']==cell['profile'] and c['snr_db']==cell['snr_db']]
            added_error = sum(c['after']['pilot_error']-c['before']['pilot_error'] for c in cases)
            added_noise = sum(c['after']['noise_var']-c['before']['noise_var'] for c in cases)
            expected_nmse = 10*np.log10((errors.sum()+added_error)/energies.sum())
            expected_noise = (noise.sum()+added_noise)/len(noise)/10**(-cell['snr_db']/10)
            assert_allclose(cell['capped_replayed_5000']['pilot_nmse_db'],expected_nmse,atol=1e-12)
            assert_allclose(cell['capped_replayed_5000']['mean_noise_ratio'],expected_noise,atol=1e-12)
            self.assertEqual(cell['capped_replayed_5000']['groups'],len(errors))
            total += len(errors)
        self.assertEqual(total,2700)

    def test_replay_and_stopping_evidence(self):
        for case in self.report['cases']:
            self.assertTrue(case['replay_check']['support_history_exact_match'])
            self.assertEqual(case['replay_check']['pilot_error_absolute_difference'],0.)
            self.assertEqual(case['replay_check']['noise_var_absolute_difference'],0.)
            self.assertEqual(case['before']['iterations'],500)
            self.assertFalse(case['before']['converged'])
            self.assertTrue(500 < case['after']['iterations'] <= 5000)
            self.assertIsNone(case['execution_failure'])
            if not case['after']['converged']:
                self.assertEqual(case['after']['iterations'],5000)
        self.assertEqual(self.report['converged_after'],2699)
        self.assertEqual(sum(not c['after']['converged'] for c in self.report['cases']),1)


if __name__=='__main__':
    unittest.main()
