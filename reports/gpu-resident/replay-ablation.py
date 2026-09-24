from pathlib import Path
p=Path('reports/gpu-resident/run_chat.py');s=p.read_text().replace('import json, subprocess, sys, time, urllib.request', 'import json, os, subprocess, sys, time, urllib.request');s=s.replace("command[command.index('--port')+1]='1923'", """command[command.index('--port')+1]='1923'
env=os.environ.copy()
if label.startswith('eager-graphs'):
    command[0]=str(folder/'llama-server-graphs.exe')
    env['GGML_CUDA_DISABLE_GRAPHS']='1'
else:
    env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
(folder/f'{label}-environment.json').write_text(json.dumps({'GGML_CUDA_DISABLE_GRAPHS':env.get('GGML_CUDA_DISABLE_GRAPHS')},indent=2))""");s=s.replace('p=subprocess.Popen(command,stdout=log', 'p=subprocess.Popen(command,env=env,stdout=log');p.write_text(s)
p=Path('reports/gpu-resident/summarize.py');s=p.read_text().replace("(folder/f'llama-server-{name}.exe').read_bytes()", "(folder/f'llama-server-{name if phase==\"boundary\" else \"graphs\"}.exe').read_bytes()");p.write_text(s)
