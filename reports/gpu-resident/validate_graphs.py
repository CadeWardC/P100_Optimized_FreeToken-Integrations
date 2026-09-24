"""Run GPU boundary/replay regressions against the local CUDA build."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'reports/gpu-resident'
bin=root/'build/ft-windows-cuda-dev/bin'
def run(label,command,env):
    print('RUN',label,flush=True)
    with (folder/f'{label}.log').open('w') as log:
        p=subprocess.run([str(x) for x in command],env=env,stdout=log,stderr=subprocess.STDOUT)
    print(label,p.returncode,flush=True)
    if p.returncode: raise SystemExit(p.returncode)
env=os.environ.copy()
for key in ('GGML_CUDA_DISABLE_GRAPHS','FT_TEST_GPU_LAYERS','FT_TEST_DEVICE_BOUNDARY','FT_TEST_REQUIRE_GRAPHS'):env.pop(key,None)
run('graphs-cpu',[bin/'test-moe-offload.exe'],env)
run('graphs-cache',[bin/'test-moe-cache.exe'],env)
env['FT_TEST_REQUIRE_GRAPHS']='1'
run('graphs-device',[bin/'test-moe-offload.exe','--device','CUDA0'],env)
run('graphs-memcheck',[Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/compute-sanitizer/compute-sanitizer.exe'),'--tool','memcheck','--error-exitcode','1',bin/'test-moe-offload.exe','--device','CUDA0'],env)
env.pop('FT_TEST_REQUIRE_GRAPHS')
run('graphs-integration',[bin/'test-moe-integration.exe','--device','CUDA0'],env)
env['FT_TEST_GPU_LAYERS']='1'
run('graphs-integration-native',[bin/'test-moe-integration.exe','--device','CUDA0'],env)
env['FT_TEST_DEVICE_BOUNDARY']='1';env['FT_TEST_REQUIRE_GRAPHS']='1'
run('graphs-pretrained-native',[bin/'test-moe-integration.exe','--device','CUDA0','--model',root/'models/gemma-4-26B_q4_0-it.gguf'],env)
env.pop('FT_TEST_GPU_LAYERS')
run('graphs-pretrained-compatibility',[bin/'test-moe-integration.exe','--device','CUDA0','--model',root/'models/gemma-4-26B_q4_0-it.gguf'],env)
print('PASS all graph validation',flush=True)
