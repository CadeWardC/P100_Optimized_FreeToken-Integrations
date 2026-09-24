import hashlib,json,statistics,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
phase=sys.argv[1] if len(sys.argv)>1 else 'boundary'
groups=([('before',['before','before-repeat']),('eager',['eager','eager-repeat'])] if phase=='boundary' else [('eager',['eager-graphs-before','eager-graphs-after']),('graphs',['graphs','graphs-repeat'])])
rows=[]; texts=[]
for engine,labels in groups:
    runs=[r for label in labels for r in json.loads((folder/f'{label}-chat.json').read_text())['runs'] if not r['warmup']]
    rates=[r['approx_decode_tps'] for r in runs]
    rows.append(dict(engine=engine,requests=len(runs),median_decode_tps=statistics.median(rates),decode_range=[min(rates),max(rates)],median_first_text_seconds=statistics.median(r['first_text_seconds'] for r in runs),median_wall_seconds=statistics.median(r['wall_seconds'] for r in runs)))
    texts.append([''.join(str(c.get('delta',{}).get(k) or '') for e in r['events'] for c in e['data'].get('choices',[]) for k in ['content','reasoning_content','reasoning']) for r in runs])
summary=dict(phase=phase,rows=rows,speed_ratio=rows[1]['median_decode_tps']/rows[0]['median_decode_tps'],matching_texts=[a==b for a,b in zip(*texts)],method='AB/BA, one excluded warmup and three measured requests per session. No concurrent build or GPU test.',executables={name:hashlib.sha256((folder/f'llama-server-{name if phase=="boundary" else "graphs"}.exe').read_bytes()).hexdigest() for name,_ in groups})
(folder/f'{phase}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
