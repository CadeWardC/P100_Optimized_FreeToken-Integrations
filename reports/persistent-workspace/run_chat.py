import json, subprocess, sys, time, urllib.request
from pathlib import Path
root=Path.cwd()
label=sys.argv[1]
folder=root/'reports/persistent-workspace'
command=json.loads((root/'reports/freetoken-installed-comparison/project-chat-command.json').read_text())
if label.startswith('before'): command[0]=str(folder/'llama-server-before.exe')
command[command.index('--port')+1]='1923'
(folder/f'{label}-command.json').write_text(json.dumps(command,indent=2))
with (folder/f'{label}-server.log').open('w') as log:
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for i in range(240):
            if p.poll() is not None: raise RuntimeError('server exited')
            try:
                urllib.request.urlopen('http://127.0.0.1:1923/health',timeout=1)
                break
            except Exception: time.sleep(1)
        else: raise RuntimeError('server timeout')
        subprocess.run([sys.executable,str(root/'scripts/benchmark_openai_stream.py'),'--base-url','http://127.0.0.1:1923','--output',str(folder/f'{label}-chat.json')],check=True)
    finally:
        p.terminate()
        p.wait(timeout=30)

