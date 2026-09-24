from pathlib import Path
p=Path('reports/persistent-workspace/run_chat.py')
s=p.read_text().replace("reports/persistent-workspace", "reports/gpu-resident")
s=s.replace("if label.startswith('before'): command[0]=str(folder/'llama-server-before.exe')", "command[0]=str(folder/('llama-server-before.exe' if label.startswith('before') else 'llama-server-eager.exe' if label.startswith('eager') else 'llama-server-graphs.exe'))")
Path('reports/gpu-resident/run_chat.py').write_text(s)
