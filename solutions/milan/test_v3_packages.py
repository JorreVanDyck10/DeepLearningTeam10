"""Check packaged candidates against saved export references and the real API."""
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
if (HERE/'.generated/preview_packages').exists():
    sys.path.insert(0,str(HERE/'.generated/preview_packages'))
import numpy as np
import pandas as pd
from mushroom_alternative import development_data


class PackageChecks(unittest.TestCase):
    def test_published_raw_and_policy_metrics_match_the_evaluated_procedures(self):
        summary=pd.read_csv(HERE/'results_v3/summary.csv').set_index('model')
        specifications=[('candidate_package.json','artifacts_v3/mushroom_rf300.json','rf_selection'),
            ('voting_package.json','voting_artifacts_v3/voting.json','voting')]
        for receipt_file,manifest,family in specifications:
            receipt=json.loads((HERE/'results_v3'/receipt_file).read_text())
            if not receipt['packaged']:
                continue
            m=json.loads((HERE/manifest).read_text(encoding='utf-8'))
            raw=summary.loc[family+'_default']; policy=summary.loc[family+'_'+receipt['policy']]
            self.assertAlmostEqual(m['metrics']['macro_f1'],raw.macro_f1_mean,places=12)
            self.assertAlmostEqual(m['metrics']['recall_poisonous'],raw.recall_p_mean,places=12)
            self.assertEqual(m['metrics']['fn'],raw.fn_total)
            self.assertAlmostEqual(m['proposed_policy_metrics']['macro_f1_mean'],policy.macro_f1_mean,places=12)
            self.assertEqual(m['proposed_policy_metrics']['fn_total'],policy.fn_total)
            self.assertEqual(m['proposed_policy_metrics']['fp_total'],policy.fp_total)
            self.assertEqual(m['test_rows'],0)

    def check_export(self,model,folder,probability_column):
        X,y,ids,_=development_data()
        expected=pd.read_csv(folder/'training_input_export_references.csv',float_precision='round_trip').set_index('source_row').loc[ids]
        actual=model.predict_proba(X)[:,1]
        np.testing.assert_allclose(actual,expected[probability_column],rtol=0,atol=1e-12)
        self.assertEqual(len(model.feature_names_in_),11)
        self.assertEqual(model.classes_.tolist(),[0,1])
        # Changing input order must not silently change the meaning of a field.
        with self.assertRaises(ValueError):
            model.predict_proba(X[X.columns[::-1]])
        record={f:None for f in X.columns}
        self.assertTrue(np.isfinite(model.predict_record(record)['probability_poisonous']))
        record['habitat']='wrong_code'
        with self.assertRaises(ValueError):
            model.predict_record(record)

    def test_rf_export(self):
        receipt=json.loads((HERE/'results_v3/candidate_package.json').read_text())
        if not receipt['packaged']:
            self.skipTest(receipt['reason'])
        from backend.mushroom_model import PortableMushroom
        self.check_export(PortableMushroom(HERE/'artifacts_v3/mushroom_rf300.json'),HERE/'artifacts_v3','probability_p')

    def test_voting_export(self):
        receipt=json.loads((HERE/'results_v3/voting_package.json').read_text())
        if not receipt['packaged']:
            self.skipTest(receipt['reason'])
        from voting_model_v3 import VotingMushroom
        self.check_export(VotingMushroom(HERE/'voting_artifacts_v3/voting.json'),HERE/'voting_artifacts_v3','probability_p')

    def check_api(self,factory,policy_id):
        from fastapi.testclient import TestClient
        with TestClient(factory()) as client:
            m=client.get('/metadata.json').json()
            self.assertEqual(len(m['fields']),11)
            self.assertEqual(m['decision_policy_id'],policy_id)
            payloads=[m['example'],{f['name']:None for f in m['fields']}]
            for payload in payloads:
                response=client.post('/predict/mushroom',json=payload)
                self.assertEqual(response.status_code,200,response.text)
                p=response.json(); score=p['probability_poisonous']; threshold=m['decision_threshold']
                self.assertEqual(p['model_revision'],m['source_pipeline_sha256'])
                self.assertEqual(p['decision_threshold'],threshold)
                self.assertEqual(p['decision_policy_id'],policy_id)
                self.assertEqual(p['label'],'p' if score>.5 else 'e')
                self.assertEqual(p['decision'],'suspected_poisonous' if score>=threshold else 'uncertain')
            bad=m['example'].copy(); bad['habitat']='wrong_code'
            self.assertEqual(client.post('/predict/mushroom',json=bad).status_code,422)
            self.assertEqual(client.post('/predict/mushroom',json={}).status_code,422)
            self.assertEqual(client.get('/health').json()['mushroom_model_revision'],m['source_pipeline_sha256'])
            html=client.get('/').text
            self.assertIn('Geen nieuwe onafhankelijke eindtest',html)
            if 'voting' in m['model_name'].lower():
                self.assertIn('via gewogen voting',html)
                self.assertNotIn('De geselecteerde Random Forest heeft 300 bomen',html)
            else:
                self.assertIn('interne featureselectie',html)

    def test_rf_api(self):
        receipt=json.loads((HERE/'results_v3/candidate_package.json').read_text())
        if not receipt['packaged']:
            self.skipTest(receipt['reason'])
        from preview_v3 import create_app,POLICY_ID
        self.check_api(create_app,POLICY_ID)

    def test_voting_api(self):
        receipt=json.loads((HERE/'results_v3/voting_package.json').read_text())
        if not receipt['packaged']:
            self.skipTest(receipt['reason'])
        from preview_voting_v3 import create_app,POLICY_ID
        self.check_api(create_app,POLICY_ID)


if __name__=='__main__':
    unittest.main(verbosity=2)
