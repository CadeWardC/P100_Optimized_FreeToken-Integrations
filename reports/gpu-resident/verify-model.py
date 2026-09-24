import hashlib,json
from pathlib import Path
root=Path.cwd()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
m=next(x for x in json.loads((root/'MODEL_ACCEPTANCE_MANIFEST.json').read_text())['models'] if x['id']=='gemma26b-qat-q4')
p=root/'models/gemma-4-26B_q4_0-it.gguf'
h=sha(p)
assert h==m['publisher_sha256'] and p.stat().st_size==m['size_bytes']
(root/'reports/gpu-resident/model-verification.json').write_text(json.dumps(dict(path=str(p),sha256=h,size_bytes=p.stat().st_size,manifest_entry=m),indent=2)+'\n')
print(h)

