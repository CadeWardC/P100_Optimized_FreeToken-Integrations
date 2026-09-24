import datetime,hashlib,json,re
from pathlib import Path
root=Path.cwd();folder=root/'reports/gpu-resident'
p=root/'README.md';s=p.read_text();a=s.index('Latest execution work:');b=s.index('\n\n',a);s=s[:a]+'Latest execution work: [GPU-resident expert decode and CUDA graph replay](GPU_RESIDENT_EXPERTS.md) keeps single-token GPU-only expert activations, routing weights and merging on device. Correctness, cancellation, cache lifecycle and CUDA replay checks pass. Replay measured 20.824 versus 18.020 tokens/s (+15.6%) with the same binary; the separate boundary-only comparison measured -4.4%. See the report for methodology and scope.'+s[b:];p.write_text(s)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
files=['llama.cpp/src/llama-moe-offload.cpp','llama.cpp/src/llama-moe-offload.h','llama.cpp/src/llama-context.cpp','llama.cpp/include/llama.h','llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu','llama.cpp/ggml/src/ggml-cuda/common.cuh','llama.cpp/tests/test-moe-offload.cpp','llama.cpp/tests/test-moe-integration.cpp','scripts/build-cuda-dev.cmd','scripts/build-cuda-graphs.cmd']
assert 'PASS all graph validation' in (folder/'validation.log').read_text()
results={}
for mode in ('native','compatibility'):
    log=(folder/f'graphs-pretrained-{mode}.log').read_text()
    maximum=max(float(x) for x in re.findall(r'GPU logits position=\d+ max_abs=([\deE.+-]+)',log))
    assert maximum==0 and 'PASS pretrained' in log
    results[mode]={'maximum_logit_difference':maximum,'logit_rows':len(re.findall('GPU logits position=',log)),'passed':True}
assert 'ERROR SUMMARY: 0 errors' in (folder/'graphs-memcheck.log').read_text()
for phase in ('boundary','replay'):
    assert all(json.loads((folder/f'{phase}-summary.json').read_text())['matching_texts'])
cache=(root/'build/ft-windows-cuda-dev/CMakeCache.txt').read_text()
assert 'GGML_CUDA_GRAPHS:BOOL=ON' in cache
binary=sha(root/'build/ft-windows-cuda-dev/bin/llama-server.exe')
assert binary==sha(folder/'llama-server-graphs.exe')
report={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':{f:sha(root/f) for f in files},'server_sha256':binary,'graph_build_enabled':True,'pretrained':results,'memcheck_errors':0,'synthetic_comparisons_per_mode':135,'source_scope':'Source hashes include pre-existing local modifications; no commit or publish performed.'}
(folder/'verification-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
