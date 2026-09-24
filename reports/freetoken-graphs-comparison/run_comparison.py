import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path.cwd();folder=root/'reports/freetoken-graphs-comparison'
project=json.loads((root/'reports/freetoken-installed-comparison/project-chat-command.json').read_text())
project[0]=str(root/'build/ft-windows-cuda-dev/bin/llama-server.exe')
project[project.index('--port')+1]='1923'
ft=[str(Path('C:/Users/cadel/AppData/Local/FreeToken/venv/Scripts/ft.exe')),'serve','--model-path',str(root/'models/gemma-4-26B_q4_0-it.gguf'),'--host','127.0.0.1','--port','1920','--moe-cache-size','959','--num-tokens','8232','--max-running-requests','1']
env=os.environ.copy();env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
for label,command,port in [('project-a',project,1923),('freetoken-a',ft,1920),('freetoken-b',ft,1920),('project-b',project,1923)]:
    if (folder/f'{label}-chat.json').exists():
        print('SKIP completed',label,flush=True)
        continue
    print('START',label,flush=True)
    (folder/f'{label}-command.json').write_text(json.dumps(command,indent=2))
    begin=time.perf_counter()
    with (folder/f'{label}-server.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(360):
                if p.poll() is not None:raise RuntimeError(f'{label} exited {p.returncode}')
                try:
                    health=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1))
                    if health.get('status')=='ok':break
                except Exception:pass
                time.sleep(1)
            else:raise TimeoutError(label)
            (folder/f'{label}-health.json').write_text(json.dumps(dict(health=health,startup_seconds=time.perf_counter()-begin),indent=2))
            print('READY',label,flush=True)
            subprocess.run([sys.executable,str(root/'scripts/benchmark_openai_stream.py'),'--base-url',f'http://127.0.0.1:{port}','--output',str(folder/f'{label}-chat.json')],check=True,env=env)
        finally:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            p.wait(timeout=30)
    print('DONE',label,flush=True)
