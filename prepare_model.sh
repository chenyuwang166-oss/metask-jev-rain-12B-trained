#!/usr/bin/env bash
# NETWORKED operator preparation, never called by serve.sh. Not executed in preparation task.
# Downloads the pinned HF revision: int4 base files at the folder root, the unmerged LoRA into adapter/.
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "${PYTHON:-python3}" -B - "$PKG" "${1:?new absolute external model directory}" <<'PY'
import json,re,sys
from pathlib import Path
pkg=Path(sys.argv[1]).resolve(); dest=Path(sys.argv[2])
m=json.loads((pkg/'MODEL_MANIFEST.json').read_text())
if not dest.is_absolute():raise SystemExit('Use an absolute destination')
dest=dest.resolve()
if dest.exists() or dest.is_relative_to(pkg) or pkg.is_relative_to(dest):raise SystemExit('Use a new separate model directory')
if m['repo_id'].startswith('OWNER/') or not re.fullmatch('[0-9a-f]{40}',str(m.get('revision') or '')):
    raise SystemExit('Set the own Hugging Face repository and a complete 40-character revision in MODEL_MANIFEST.json')
if not m['files']:raise SystemExit('Model hash manifest is pending')
patterns=[*m['files'],*m.get('audit_files',{})]
adapter=m.get('adapter')
if adapter is not None:
    if not adapter.get('files'):raise SystemExit('Adapter hash manifest is pending')
    patterns+=[adapter['subdir']+'/'+name for name in adapter['files']]
from huggingface_hub import snapshot_download
snapshot_download(repo_id=m['repo_id'],revision=m['revision'],local_dir=dest,allow_patterns=patterns)
sys.path.insert(0,str(pkg))
from launch import check_model
check_model(dest,full=True)
print('Downloaded base and adapter files match the pinned manifest; no inference or submission performed.')
PY
