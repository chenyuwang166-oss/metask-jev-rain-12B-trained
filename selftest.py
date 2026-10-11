"""Synthetic offline selftest. Never imports a model runtime or opens a socket."""
import argparse
import copy
import hashlib
import json
import math
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import core
import launch
import pkg_check

class SelfTest(unittest.TestCase):
    def test_package_integrity(self):
        self.assertIn(pkg_check.checks()['status'],('PREPARATION_CHECK_PASS','FROZEN_PACKAGE_CHECK_PASS'))

    def test_pending_cannot_serve(self):
        config=pkg_check.read('config.json')
        if not config['package'].get('nonformal_fitted'):
            with self.assertRaisesRegex(ValueError,'fit pending'):
                launch.check_preparation(allow_nonformal=True)
        if config['package'].get('finalized'):
            launch.check_preparation()  # a frozen package starts formally, without any extra switch
        else:
            with self.assertRaises(ValueError):launch.check_preparation()

    def test_offline_environment(self):
        base=Path.cwd().anchor
        state=Path(base)/'jev-v15-offline-state'
        env=launch.child_environment({},state,state/'startup')
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','VLLM_BATCH_INVARIANT'):
            self.assertEqual(env[key],'1')
        self.assertEqual(env['VLLM_USE_FLASHINFER_SAMPLER'],'0')
        self.assertTrue(Path(env['TMPDIR']).is_relative_to(state))

    def test_missing_model_fails_closed(self):
        if not launch._MANIFEST['files']:
            with self.assertRaisesRegex(ValueError,'identity pending'):
                launch.check_model(Path.cwd())

    def test_threshold_identity(self):
        c=pkg_check.read('config.json')
        expected=core.threshold_identity(c)
        self.assertEqual(expected['model_key'],pkg_check.MODEL_KEY)
        self.assertEqual(expected['prompt_sha'],'e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938')
        c['routing']['margin_threshold_fitted_for']={'placeholder':False,**expected}
        self.assertEqual(core.validate_threshold(c),[])
        c['routing']['margin_threshold_fitted_for']['model_key']='wrong'
        with self.assertRaises(ValueError):core.validate_threshold(c)

    def test_adapter_reference_materializes_into_served_identity(self):
        c=pkg_check.read('config.json')
        identity=core.threshold_identity(c)
        c['routing']['margin_threshold_fitted_for']={'placeholder':False,**identity}
        c['profiles'][pkg_check.MODEL_KEY]['routing']=copy.deepcopy(c['routing'])
        folder=Path(tempfile.gettempdir())/'synthetic-model'/'adapter'
        if c['model'].get('adapter') is None:
            self.assertEqual(launch.materialize_adapter(c,None),c)
            with self.assertRaises(ValueError):launch.materialize_adapter(c,folder)
            profiled=copy.deepcopy(c);profiled['profiles'][pkg_check.MODEL_KEY]['model']['adapter']='/elsewhere'
            with self.assertRaises(ValueError):launch.materialize_adapter(profiled,None)
            return
        self.assertEqual(c['model']['adapter'],launch.ADAPTER_REF)
        served=launch.materialize_adapter(c,folder)
        self.assertEqual(served['model']['adapter'],str(folder))
        self.assertEqual(served['routing']['margin_threshold_fitted_for']['adapter'],str(folder))
        self.assertEqual(core.validate_threshold(served),[])
        self.assertEqual(c['model']['adapter'],launch.ADAPTER_REF)
        with self.assertRaises(ValueError):launch.materialize_adapter(c,None)
        raw=copy.deepcopy(c);raw['model']['adapter']=str(folder)
        with self.assertRaises(ValueError):launch.materialize_adapter(raw,folder)
        other=copy.deepcopy(c);other['routing']['margin_threshold_fitted_for']['adapter']='/elsewhere'
        with self.assertRaises(ValueError):launch.materialize_adapter(other,folder)

    def test_adapter_folder_is_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)
            folder=model/'adapter';folder.mkdir()
            payload={'adapter_config.json':b'{"r": 16}\n','adapter_model.safetensors':b'synthetic weights'}
            pinned={}
            for name,data in payload.items():
                (folder/name).write_bytes(data)
                pinned[name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}
            with patch.object(launch,'_ADAPTER',{'files':pinned}),patch.object(launch,'ADAPTER_FILES',pinned):
                self.assertEqual(launch.check_adapter(model),folder.resolve())
                (folder/'extra.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'exactly'):launch.check_adapter(model)
                (folder/'extra.json').unlink()
                (folder/'adapter_model.safetensors').write_bytes(b'tampered weights!')
                with self.assertRaisesRegex(ValueError,'differs'):launch.check_adapter(model)
                (folder/'adapter_model.safetensors').unlink()
                with self.assertRaises(ValueError):launch.check_adapter(model)
            with patch.object(launch,'_ADAPTER',{'files':{}}),patch.object(launch,'ADAPTER_FILES',{}):
                with self.assertRaisesRegex(ValueError,'pending'):launch.check_adapter(model)
            with patch.object(launch,'_ADAPTER',None),patch.object(launch,'ADAPTER_FILES',{}):
                self.assertIsNone(launch.check_adapter(model))

    def test_readout_probability_contract(self):
        block={'temperature_by_kind':{'choice':1.,'score':1.,'noul':'one_bin'},'noul_calibration':{'p_cal':.81}}
        for kind,labels in [('choice',['A','B','C']),('score',['1','2','3']),('noul',['no','yes'])]:
            logits={label:float(i) for i,label in enumerate(labels)}
            p=core.probabilities(labels,logits,kind,block)
            self.assertEqual(set(p),set(labels))
            self.assertAlmostEqual(sum(p.values()),1.)
            self.assertTrue(all(math.isfinite(v) and 0<=v<=1 for v in p.values()))
            self.assertEqual(max(labels,key=p.__getitem__),labels[-1])
        p=core.probabilities(['no','yes'],{'no':0.,'yes':0.},'noul',block)
        self.assertGreater(p['no'],p['yes'])

    def test_readout_rejects_missing_and_nonfinite(self):
        for logits in ({0:0.},{0:0.,1:float('nan')},{0:0.,1:float('inf')}):
            with self.assertRaises(RuntimeError):core.aggregate(logits,{'a':[0],'b':[1]})

    def test_config_does_not_accept_invalid_quota(self):
        c=copy.deepcopy(pkg_check.read('config.json'))
        c['routing']['quota']=.9
        with self.assertRaises(ValueError):core.validate_config(c)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',action='store_true',help='default; compatibility flag')
    p.parse_args()
    with patch.object(socket,'socket',side_effect=AssertionError('network forbidden in offline selftest')):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
                      'synthetic':True,'model_called':False,'network':False,'gpu_selftest':'pending A100'},ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
