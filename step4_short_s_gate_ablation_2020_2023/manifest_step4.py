"""Generate and verify a self-excluding complete artifact manifest."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='artifact_manifest.json'}
(P/'artifact_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in files.items())
print('Manifest verified:',len(files),'files')
