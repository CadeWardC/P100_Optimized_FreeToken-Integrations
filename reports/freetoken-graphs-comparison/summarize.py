import hashlib,json,statistics
from pathlib import Path
root=Path.cwd();folder=root/'reports/freetoken-graphs-comparison'
rows=[]
for engine in ('project','freetoken'):
    reports=[json.loads((folder/f'{engine}-{order}-chat.json').read_text()) for order in ('a','b')]
    assert all(len(r['runs'])==4 for r in reports)
    runs=[x for r in reports for x in r['runs'] if not x['warmup']]
    rates=[x['approx_decode_tps'] for x in runs]
    texts=[''.join(str(c.get('delta',{}).get(k) or '') for e in x['events'] for c in e['data'].get('choices',[]) for k in ('content','reasoning_content','reasoning')) for x in runs]
    (folder/f'{engine}-responses.json').write_text(json.dumps(texts,indent=2))
    row=dict(engine=engine,measured_requests=len(runs),median_decode_tps=statistics.median(rates),decode_range=[min(rates),max(rates)],median_first_text_seconds=statistics.median(x['first_text_seconds'] for x in runs),median_wall_seconds=statistics.median(x['wall_seconds'] for x in runs),prompt_tokens=[x['usage']['prompt_tokens'] for x in runs],completion_tokens=[x['usage']['completion_tokens'] for x in runs],per_session_median=[statistics.median(x['approx_decode_tps'] for x in r['runs'] if not x['warmup']) for r in reports],within_engine_matching_texts=len(set(texts))==1)
    rows.append(row)
    print(engine,'sample:',texts[0][:500])
summary=dict(rows=rows,project_over_freetoken_decode_ratio=rows[0]['median_decode_tps']/rows[1]['median_decode_tps'],method='AB/BA separate engines, one excluded warmup plus three measured requests per session. Same GGUF file, chat prompt, thinking disabled and 128-token request. FreeToken fixed at 959 expert slots and 8232 KV tokens; project expert budget 3060 MiB. Prefix reuse and generated text differ.',model=json.loads((root/'reports/gpu-resident/model-verification.json').read_text()),project_server_sha256=hashlib.sha256((root/'build/ft-windows-cuda-dev/bin/llama-server.exe').read_bytes()).hexdigest(),freetoken_health=[json.loads((folder/f'freetoken-{order}-health.json').read_text()) for order in ('a','b')])
(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('model','freetoken_health')},indent=2))
