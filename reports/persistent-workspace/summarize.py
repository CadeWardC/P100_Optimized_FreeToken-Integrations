import hashlib, json, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'reports/persistent-workspace'
rows=[]
texts=[]
for engine,labels in [('before',['before-clean','before-repeat']),('after',['after','after-repeat'])]:
    runs=[r for label in labels for r in json.loads((folder/f'{label}-chat.json').read_text())['runs'] if not r['warmup']]
    rows.append(dict(engine=engine,measured_requests=len(runs),median_decode_tps=statistics.median(r['approx_decode_tps'] for r in runs),decode_range=[min(r['approx_decode_tps'] for r in runs),max(r['approx_decode_tps'] for r in runs)],median_first_text_seconds=statistics.median(r['first_text_seconds'] for r in runs),median_wall_seconds=statistics.median(r['wall_seconds'] for r in runs)))
    texts.append([''.join(str(c.get('delta',{}).get(k) or '') for e in r['events'] for c in e['data'].get('choices',[]) for k in ['content','reasoning_content','reasoning']) for r in runs])
summary=dict(rows=rows,speed_ratio=rows[1]['median_decode_tps']/rows[0]['median_decode_tps'],matching_response_texts=[a==b for a,b in zip(*texts)],scope='AB then BA: same model and launch settings; one warmup and three measured requests per session, six measured requests per executable. No concurrent builds/tests. Initial before-chat run overlapped compilation and is excluded.',conclusion='No demonstrated speedup. Updated pooled median decode is slightly lower; observed ranges overlap. Workspace allocation reuse is verified independently.')
summary['executables']={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in [('before',folder/'llama-server-before.exe'),('after',root/'build/ft-windows-cuda-dev/bin/llama-server.exe')]}
(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
