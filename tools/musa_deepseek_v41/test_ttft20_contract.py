"""Synthetic comparison fixtures; only the immutable old r0 audits are measured data."""
import copy
import json
from pathlib import Path
import unittest
from ttft20_contract import *
from test_long_20k_200k_gate import LongGoalGateTest as _Legacy

class ContractTest(unittest.TestCase):
    def setUp(self):
        r0=json.loads((Path(__file__).resolve().parents[2]/'docs/zh/long_c1_20261119/S20AB冻结参考_r0.json').read_text())
        self.b=dict(r0=r0,protocol_sha256=protocol_hash(MAIN),schema='ttft20-reference-set-v1',references={})
        self.state=dict(resume_revision=REVISION,status='ACTIVE',allowed_lengths=[20000],execution_authorized=True)
        for role in MODELS:
            runs=[]
            for i in range(2):
                rows=[];tf={'A':800,'V':1300,'S':1200}[role]
                for j in range(6):
                    id_=f'{role}.{i}.{j}'
                    rows.append(dict(phase='formal' if j else 'warmup',id=id_,terminal_id=id_,unique_terminal_count=1,
                                  success=True,prompt_tokens=20000,completion_tokens=1024,cached_tokens=0,reasoning_tokens=0,
                                  finish_reason='length',accepted=500,proposed=1500,text_sha256=digest('text'),
                                  ttft_ms=tf,tpot_ms=10.,e2e_s=(tf+10230)/1000))
                runs.append(dict(uid=f'{role}{i}',config_sha256=digest('cfg'),assets_sha256=digest('asset'),
                        released={'sj9':True,'sj10':True},compile_clear=True,profiler_on=False,
                        starts={r:dict(host='sj9' if r=='P' else 'sj10',container=digest(r),pid=i+350,start_ticks=i+200)
                                for r in ('P','D','router')},samples=rows,
                        evidence_sha256={k:digest(k) for k in ('warm_db','formal_db','terminal','launch_p','launch_d',
                        'launch_router','release_p','release_d','loaded','compile_window','identity')}))
            self.b['references'][role]=dict(model=MODELS[role],role=role,framework='sglang' if role=='S' else 'xllm',
                        protocol_sha256=protocol_hash(MAIN),version='r1',runs=runs,semantic_recipe_sha256=digest('recipe'),
                        environment_sha256=digest('env'),dataset_sha256=V41_DATA if role=='S' else digest(role),
                        token_ids_sha256=V41_IDS if role=='S' else digest(role+'ids'),tokenizer_sha256=digest(role))
    def test_roles(self):
        self.assertEqual(summary(self.b)['target'],dict(ttft_ms=1000,tpot_ms=10.,e2e_s=11.43))
        del self.b['references']['V']
        self.assertFalse(compare(self.b,dict(ttft_ms=1,tpot_ms=1,e2e_s=2))['thresholds_met'])
        with self.assertRaises(ValueError):judge(self.b,[])
    def test_reference_fail_closed(self):
        for edit in (lambda r:r.update(model=MODELS['V']),lambda r:r.update(protocol_sha256=protocol_hash('sg-guide-pd-c48')),
                     lambda r:r.update(token_ids_sha256='0'*64),lambda r:r.update(runs=[]),
                     lambda r:r['runs'][1]['starts'].update(D=r['runs'][0]['starts']['D']),
                     lambda r:r['runs'][0]['samples'][1].update(reasoning_tokens=1),
                     lambda r:r['runs'][1]['samples'][2].update(cached_tokens=5),
                     lambda r:r['runs'][0]['samples'][0].update(prompt_tokens=32000),
                     lambda r:r['runs'][0].update(released={}),lambda r:r['runs'][1].update(config_sha256='0'*64),
                     lambda r:r.update(semantic_recipe_sha256=digest('different'))):
            b=copy.deepcopy(self.b);edit(b['references']['S'])
            with self.assertRaises(ValueError): summary(b)
        self.b['r0']['raw_audits'][0]['samples'][1]['ttft_ms']+=1
        with self.assertRaises(ValueError):summary(self.b)
    def test_scope(self):
        guard(self.state,MAIN,'reference',(20000,1024,1),'A')
        guard(self.state,'sg-guide-prefill','auxiliary',(32000,1,1),'S')
        guard(self.state,RESOURCE_2K,'auxiliary',(2000,1024,1),'X')
        for prot,purpose,shape,src in [('sg-guide-prefill','formal',(32000,1,1),'S'),
                (MAIN,'reference',(32000,1000,48),'S'),('albert-original','auxiliary',(20000,1,1),'X'),
                (MAIN,'screen',(200000,1024,1),'X'),
                (MAIN,'screen',(2000,1024,1),'X'),
                (RESOURCE_2K,'screen',(2000,1024,1),'X')]:
            with self.assertRaises(ValueError):guard(self.state,prot,purpose,shape,src)
        with self.assertRaises(ValueError):guard(dict(self.state,status='PAUSED_BY_USER'),MAIN,'diagnostic',(20000,1024,1),'S')
    def test_formal_bindings(self):
        fixture=_Legacy();fixture.setUp();groups=fixture.groups
        bind=digest(self.b)
        for g in groups:
            g.update(reference_sha256=bind,protocol_sha256=protocol_hash(MAIN),
                     compilation={h:{'all_clear':True} for h in ('sj9','sj10')})
            g['identity'].update(length=20000,token_ids_sha256=V41_IDS,dataset_sha256=V41_DATA)
            g['raw_audit'].update(length=20000,files={'db':digest('db')})
            g['execution_config_sha256']=digest('cfg')
            reg=g['registration'];reg.update(length=20000,reference_sha256=bind,protocol_sha256=protocol_hash(MAIN),
                screen_config_sha256=g['execution_config_sha256'],screen_assets_sha256=digest(g['assets']))
            reg['sha256']=digest({k:v for k,v in reg.items() if k!='sha256'})
        self.assertTrue(judge(self.b,groups)['pass'])
        changed=copy.deepcopy(groups)
        for g in changed:g['assets'].update(ELF='synchronized change on BOTH F and R')
        with self.assertRaises(ValueError):judge(self.b,changed)
        groups[1]['reference_sha256']=digest('other set')
        with self.assertRaises(ValueError):judge(self.b,groups)
        groups[1]['reference_sha256']=bind
        groups[0]['released']['sj10']=False
        with self.assertRaises(ValueError):judge(self.b,groups)

if __name__=='__main__':unittest.main()
