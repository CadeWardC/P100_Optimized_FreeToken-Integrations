"""Measure an already-running local OpenAI chat endpoint; retain SSE events."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = json.load(urllib.request.urlopen(args.base_url + '/v1/models'))['data'][0]['id']
    body = dict(model=model, messages=[{'role': 'user', 'content': 'Explain how a GPU works.'}],
                max_tokens=128, temperature=0, ignore_eos=True, stream=True,
                chat_template_kwargs={'enable_thinking': False},
                stream_options={'include_usage': True})
    report = dict(request=body, runs=[],
                  timing_note='Approximate decode rate from first to last nonempty text/reasoning event; '
                  'not an internal GPU timer. Usage token counts are engine-reported.')
    for index in range(4):
        start = time.perf_counter()
        events = []
        req = urllib.request.Request(args.base_url + '/v1/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                if line.startswith(b'data: ') and b'[DONE]' not in line:
                    events.append(dict(seconds=time.perf_counter() - start,
                                       data=json.loads(line[6:])))
        elapsed = time.perf_counter() - start
        def has_text(event):
            return any(any(c.get('delta', {}).get(k) for k in
                           ('content', 'reasoning_content', 'reasoning'))
                       for c in event['data'].get('choices', []))
        texts = [e for e in events if has_text(e)]
        usage = next(e['data']['usage'] for e in reversed(events) if e['data'].get('usage'))
        if len(texts) < 2 or usage['completion_tokens'] < 2:
            raise RuntimeError('Insufficient text/token data for decode measurement')
        run = dict(warmup=index == 0, wall_seconds=elapsed, usage=usage,
                   first_text_seconds=texts[0]['seconds'], last_text_seconds=texts[-1]['seconds'],
                   text_chunks=len(texts),
                   approx_decode_tps=(usage['completion_tokens'] - 1) /
                   (texts[-1]['seconds'] - texts[0]['seconds']),
                   end_to_end_tps=usage['completion_tokens'] / elapsed, events=events)
        report['runs'].append(run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print({k: v for k, v in run.items() if k != 'events'}, flush=True)


if __name__ == '__main__':
    main()
