"""Offline package checks. --release lists what is still missing; it never approves a release.

A preparation package (config package.finalized false) and a frozen package
(finalized true plus FREEZE.json) are both checked. The freeze may change only
package.finalized in config.json and replace README.md, DISCLOSURE.md and
MODEL_CARD.md; every other file must keep the bytes of the prepared package.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
MODEL_KEY='metask-gemma4-12b-lora-awq'
ADAPTER_REF='model-subdir:adapter'
ADAPTER_SUBDIR='adapter'
ADAPTER_FILES=('adapter_config.json','adapter_model.safetensors')
LORA=dict(rank=16,alpha=32,target_modules=sorted(('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')))
BASE_MODEL='google/gemma-4-12B-it-qat-q4_0-unquantized'
BASE_REVISION='b6ed86275a6a5735884e208bfed95b445a684ca2'
LOCAL_ONLY_AUDIT=('AWQ_MANIFEST.json','QUANTIZATION_STARTED.json')
REQUIRED=('core.py backends.py quota.py attention.py model_setup.py server.py bench.py '
          'reconcile_usage.py launch.py serve.sh selftest.py selftest.sh pkg_check.py '
          'prepare_model.sh config.json calibration.json MODEL_MANIFEST.json '
          'PIPELINE_PROVENANCE.json DISCLOSURE_SOURCES.json DISCLOSURE.md README.md MODEL_CARD.md '
          'requirements.lock requirements.measured.lock LICENSE MANIFEST.sha256').split()
RUNTIME=('core.py','backends.py','quota.py','attention.py','model_setup.py','server.py','launch.py')
FREEZE_SCHEMA='submit_v15_freeze_v1'
FREEZE_DOCUMENTS=('README.md','DISCLOSURE.md','MODEL_CARD.md')
CONFIG_CHANGE='package.finalized: false -> true (single textual replacement)'
FROZEN_FLAG,PREPARED_FLAG=b'"finalized": true',b'"finalized": false'
# Release text must not carry internal working names, machine paths or template markers.
INTERNAL_MARKERS=(r'(?i)(?<![a-z])psi(?![a-z])','ψ',r'\bOpus\b',r'\bCodex\b',r'/home/',r'review_2026',
                  r'submission_v2',r'(?<![A-Za-z])Rain(?![A-Za-z])',r'[A-Za-z]:(?:\\|\\\\|/)Users')
TEMPLATE_MARKERS=('PENDING','TEMPLATE','<HF_REPO>','<HF_REVISION>')
TEXT_SUFFIXES=('.py','.md','.json','.sh','.lock','.txt')

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def identity_sha256(m):
    """Base-only identity (v14 formula) or base + unmerged adapter identity."""
    identity={'format':m['format'],'files':m['files']}
    if m.get('adapter') is not None:
        a=m['adapter']
        identity['adapter']={'format':a['format'],'subdir':a['subdir'],'files':a['files']}
    return canonical(identity)

def check_manifest(root=ROOT):
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if not re.fullmatch('[0-9a-f]{64}',digest) or Path(name).name!=name or name in expected:
            raise ValueError('Unsafe or duplicate manifest entry')
        p=root/name
        if p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Missing, linked or changed package member: '+name)
        expected[name]=digest
    actual={p.name for p in root.iterdir() if p.is_file() and p.name!='MANIFEST.sha256'}
    if actual!=set(expected):raise ValueError('Manifest membership differs')
    for p in root.iterdir():
        if p.is_symlink() or (p.is_dir() and p.name!='__pycache__'):
            raise ValueError('Unexpected package directory or symlink')
    return expected

def front_matter(text):
    """Minimal YAML front-matter reader for top-level 'key: value' lines."""
    lines=text.splitlines()
    if not lines or lines[0].strip()!='---':raise ValueError('Model card lacks YAML front matter')
    values={}
    for line in lines[1:]:
        if line.strip()=='---':return values
        match=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*):\s*(\S.*)?',line)
        if match:values[match[1]]=(match[2] or '').strip()
    raise ValueError('Unterminated model card front matter')

def check_adapter_contract(c,m):
    arm=c['pipeline']['arm']
    adapter=m.get('adapter')
    profiles=[(p.get('model') or {}) for p in c.get('profiles',{}).values()]
    if arm=='FINALIST':
        if (not isinstance(adapter,dict) or adapter.get('format')!='peft-lora' or adapter.get('subdir')!=ADAPTER_SUBDIR
                or not isinstance(adapter.get('files'),dict)):
            raise ValueError('FINALIST manifest must pin the unmerged adapter/ folder')
        if adapter['files'] and (set(adapter['files'])!=set(ADAPTER_FILES) or adapter.get('merged') is not False
                or adapter.get('rank')!=LORA['rank'] or adapter.get('alpha')!=LORA['alpha']
                or adapter.get('target_modules')!=LORA['target_modules']
                or adapter.get('base_model')!=BASE_MODEL or adapter.get('base_revision')!=BASE_REVISION):
            raise ValueError('Adapter manifest differs from the trained r16/alpha32 unmerged LoRA contract')
        if c['model'].get('adapter')!=ADAPTER_REF or any('adapter' in p and p['adapter']!=ADAPTER_REF for p in profiles):
            raise ValueError('FINALIST config must reference the model-relative adapter')
        rank=c['model'].get('vllm_options',{}).get('max_lora_rank')
        if type(rank) is not int or rank<LORA['rank']:
            raise ValueError('vLLM max_lora_rank must cover the adapter rank')
        if 'enable_lora' in c['model'].get('vllm_options',{}):
            raise ValueError('enable_lora is backend-owned and must not be configured')
    else:
        # A profile value would override the top level when the server applies the profile.
        if adapter is not None or c['model'].get('adapter') is not None or any(p.get('adapter') is not None for p in profiles):
            raise ValueError('U0 package must not carry an adapter')
    local=m.get('local_audit_files') or {}
    if set(local)-set(LOCAL_ONLY_AUDIT) or set(local)&set(m.get('audit_files') or {}):
        raise ValueError('Only the quantizer records may stay local, and never also published')

def prepared_config_bytes(data):
    """Undo the freeze's single textual change of config.json."""
    if data.count(FROZEN_FLAG)!=1 or PREPARED_FLAG in data:
        raise ValueError('Frozen config.json must differ from the prepared one only in package.finalized')
    return data.replace(FROZEN_FLAG,PREPARED_FLAG)

def check_freeze(c,m,members):
    """None for a preparation package; the verified FREEZE.json record for a frozen one."""
    path=ROOT/'FREEZE.json'
    if not c['package'].get('finalized'):
        if path.exists():raise ValueError('FREEZE.json in a package that is not finalized')
        return None
    if not path.is_file():raise ValueError('Finalized package without FREEZE.json')
    f=read('FREEZE.json')
    if f.get('schema')!=FREEZE_SCHEMA or f.get('protocol')!='submit_v15' or f.get('status')!='FROZEN':
        raise ValueError('Unknown freeze record')
    if c['pipeline']['arm']!='FINALIST' or not c['package'].get('nonformal_fitted') or c['package'].get('sample_only'):
        raise ValueError('Only the fitted FINALIST package can be frozen')
    if not re.fullmatch('[0-9a-f]{40}',str(m.get('revision') or '')) or m['repo_id'].startswith('OWNER/'):
        raise ValueError('Frozen package lacks the own repository and full revision')
    if (f.get('repo_id'),f.get('revision'),f.get('model_identity_sha256'))!=(m['repo_id'],m['revision'],m['identity_sha256']):
        raise ValueError('Freeze record names another repository, revision or model')
    prepared=f.get('prepared_files')
    replaced=f.get('replaced_documents') or {}
    if not isinstance(prepared,dict) or set(members)-{'FREEZE.json'}!=set(prepared) or set(replaced)-set(FREEZE_DOCUMENTS):
        raise ValueError('Frozen package membership differs from the prepared package')
    for name,digest in members.items():
        if name=='FREEZE.json':continue
        if name=='config.json':
            before=hashlib.sha256(prepared_config_bytes((ROOT/name).read_bytes())).hexdigest()
            if before!=prepared[name] or f.get('config')!={'prepared_sha256':before,'frozen_sha256':digest,'change':CONFIG_CHANGE}:
                raise ValueError('Frozen config.json is not the recorded freeze of the prepared config')
        elif name in replaced:
            if replaced[name]!={'prepared_sha256':prepared[name],'frozen_sha256':digest}:
                raise ValueError('Replaced document differs from the freeze record: '+name)
        elif digest!=prepared[name]:
            raise ValueError('Freeze changed a file other than config.json and the documents: '+name)
    return f

def internal_markers(root=ROOT):
    """Internal working names, machine paths in package text files (pkg_check.py itself excluded)."""
    repo=json.loads((root/'MODEL_MANIFEST.json').read_text(encoding='utf-8')).get('repo_id') or ''
    hits=[]
    for p in sorted(root.iterdir()):
        if not p.is_file() or p.name=='pkg_check.py' or not (p.suffix in TEXT_SUFFIXES or p.name in ('LICENSE','.gitignore','.gitattributes')):
            continue
        text=p.read_text(encoding='utf-8',errors='replace').replace(repo,'')
        for pattern in INTERNAL_MARKERS:
            if re.search(pattern,text):hits.append(p.name+': '+pattern)
    return hits

def template_markers(root=ROOT):
    hits=[]
    for name in FREEZE_DOCUMENTS:
        text=(root/name).read_text(encoding='utf-8',errors='replace')
        hits.extend(name+': '+marker for marker in TEMPLATE_MARKERS if marker in text)
    return hits

def checks():
    for name in REQUIRED:
        if not (ROOT/name).is_file():raise ValueError('Missing '+name)
    members=check_manifest()
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    c,m,prov=read('config.json'),read('MODEL_MANIFEST.json'),read('PIPELINE_PROVENANCE.json')
    from core import validate_config, validate_threshold, validate_calibration
    import launch
    validate_config(c)
    if c['model']['model_key']!=MODEL_KEY or set(c['profiles'])!={MODEL_KEY}:
        raise ValueError('Wrong trained model key')
    if c['pipeline']['protocol']!='submit_v15' or c['pipeline']['arm'] not in ('FINALIST','U0'):
        raise ValueError('Wrong protocol/arm')
    if c['pipeline']['arm']!=prov.get('arm') or m.get('arm')!=c['pipeline']['arm']:
        raise ValueError('Arm differs between config, manifest and provenance')
    if c['pipeline']['model_identity_sha256']!=m['identity_sha256']:
        raise ValueError('Model identity mismatch')
    if m.get('revision') is not None and not re.fullmatch('[0-9a-f]{40}',m['revision']):
        raise ValueError('Abbreviated/mutable revision rejected')
    if m['repo_id'].startswith(('google/','cyankiwi/')):
        raise ValueError('Own HF repository required')
    check_adapter_contract(c,m)
    frozen=check_freeze(c,m,members)
    complete=bool(m['files']) and (m.get('adapter') is None or bool(m['adapter']['files']))
    if complete and identity_sha256(m)!=m['identity_sha256']:raise ValueError('Manifest identity hash differs')
    if not complete and m['identity_sha256'] is not None:raise ValueError('Incomplete model has an identity')
    cal=read('calibration.json')
    if set(cal['models'])!={MODEL_KEY}:raise ValueError('Foreign calibration model')
    for side in ('fast','slow','fallback'):validate_calibration(cal,MODEL_KEY,side)
    fitted=c['package'].get('nonformal_fitted',False)
    if fitted:
        validate_threshold(c)
        # The served configuration is the materialized one; its threshold identity must still hold.
        validate_threshold(launch.materialize_adapter(c,Path('/materialized/adapter') if m.get('adapter') else None))
        lock=read('SERVICE_LOCK.json')
        report=read('fit_report.json')
        if (lock['protocol']!='submit_v15' or lock['model_identity_sha256']!=m['identity_sha256']
                or lock['arm']!=c['pipeline']['arm']
                or lock.get('adapter_files')!=((m.get('adapter') or {}).get('files') if m.get('adapter') else None)):
            raise ValueError('Fit lock identity mismatch')
        actual={'config':frozen['config']['prepared_sha256'] if frozen else sha(ROOT/'config.json'),
                'calibration':sha(ROOT/'calibration.json'),'fit_report':sha(ROOT/'fit_report.json')}
        for stem,digest in actual.items():
            if lock[stem+'_sha256']!=digest:raise ValueError('Fit hash mismatch')
        if report.get('fit_protocol')!='original-12b-public231-v1.1':raise ValueError('Wrong fitting method')
    else:
        if not c['package']['sample_only'] or not cal['models'][MODEL_KEY].get('placeholder'):
            raise ValueError('Unfitted package must clearly carry placeholders')
        if not validate_threshold(c,allow_placeholder=True):raise ValueError('Unfitted threshold disguised as fitted')
    if (c['package']['finalized'] and frozen is None) or prov.get('release_ready') or prov.get('official_submission_authorized'):
        raise ValueError('Only the recorded freeze may finalize; the builder never authorizes a release')
    for name in ('core.py','backends.py','quota.py','attention.py','model_setup.py','server.py','bench.py','reconcile_usage.py'):
        if sha(ROOT/name)!=prov['source_files'][name]:raise ValueError('F request code changed: '+name)
    if c['routing']['quota']!=.15 or c['routing']['fuse']!=.20 or c['usage']['basis']!='engine':
        raise ValueError('Changed F serving/cost convention')
    for name in RUNTIME:
        text=(ROOT/name).read_text(encoding='utf-8')
        if re.search(r'\brmtree\s*\(|\b(?:snapshot_download|hf_hub_download)\s*\(',text):
            raise ValueError('Runtime downloads or recursive deletion found')
    launch_text=(ROOT/'launch.py').read_text(encoding='utf-8')
    for token in ("HF_HUB_OFFLINE='1'","TRANSFORMERS_OFFLINE='1'","env['VLLM_BATCH_INVARIANT']='1'",
                  "VLLM_USE_FLASHINFER_SAMPLER='0'","def check_adapter(","def materialize_adapter(",
                  "    check_adapter(model)\n","cfg=write_runtime_config(startup,runtime)","config.runtime.json",
                  "LOCAL_AUDIT_SHA256","(ROOT/'FREEZE.json').is_file()"):
        if token not in launch_text:raise ValueError('Offline/invariant/adapter/freeze startup setting missing: '+token.strip())
    if launch.ADAPTER_REF!=ADAPTER_REF:raise ValueError('Launcher adapter reference differs')
    for p in ROOT.glob('*.sh'):
        if re.search(r'\brm\s+-[^\n]*r|\brmtree\b',p.read_text(encoding='utf-8')):
            raise ValueError('Recursive deletion in shell')
    if "VLLM_USE_FLASHINFER_SAMPLER=0" not in (ROOT/'serve.sh').read_text(encoding='utf-8'):
        raise ValueError('serve.sh must disable the FlashInfer sampler like F')
    card=front_matter((ROOT/'MODEL_CARD.md').read_text(encoding='utf-8'))
    if card.get('base_model')!=BASE_MODEL:
        raise ValueError('Model card front matter must name base_model '+BASE_MODEL)
    pins=dict(line.split('==',1) for line in (ROOT/'requirements.lock').read_text().splitlines() if '==' in line and not line.startswith('#'))
    for name,version in launch.VERSIONS.items():
        if pins.get(name,'').split('+')[0]!=version:raise ValueError('Runtime version pin mismatch')
    return {'status':'FROZEN_PACKAGE_CHECK_PASS' if frozen else 'PREPARATION_CHECK_PASS','model_execution':False,'network':False,
            'arm':c['pipeline']['arm'],'adapter':'unmerged adapter/' if m.get('adapter') else None,
            'service_fit':'frozen' if frozen else 'fitted_not_frozen' if fitted else 'pending','release_ready':False}

def release_problems():
    c,m=read('config.json'),read('MODEL_MANIFEST.json')
    problems=[]
    if not c['package'].get('nonformal_fitted'):problems.append('arm-specific public231 fit pending')
    if not m['files']:problems.append('local U0 AWQ base file hashes pending')
    if m.get('adapter') is not None and not m['adapter'].get('files'):problems.append('adapter/ file hashes pending')
    if m['repo_id'].startswith('OWNER/') or not re.fullmatch('[0-9a-f]{40}',str(m.get('revision') or '')):
        problems.append('own Hugging Face repository and complete revision pending')
    if not c['package'].get('finalized'):
        problems.extend(['A100 service cost/latency, zero-error and adapter-effect evidence pending review',
                         'A100 reproducibility evidence and batch-invariant decision pending review',
                         'A100 package-launch smoke of this exact package pending review',
                         'freeze with freeze_package_v15.py pending (formal start needs FREEZE.json)'])
    problems.extend(template_markers())
    problems.extend('internal marker: '+hit for hit in internal_markers())
    problems.extend(['confirmation cold run: download the published revision and start strictly as README.md describes',
                     'upload and submission authorization by the repository owner'])
    return problems

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',action='store_true')
    a=p.parse_args()
    result=checks()
    if a.release:
        result.update(status='RELEASE_NOT_READY',missing=release_problems())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2 if a.release else 0

if __name__=='__main__':raise SystemExit(main())
