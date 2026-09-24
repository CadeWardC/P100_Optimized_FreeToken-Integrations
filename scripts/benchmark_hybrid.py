"""Benchmark local hybrid miss policies through matched server requests.

Requires psutil. Warm-up is excluded; expert cache persists between requests,
while prompt KV reuse is disabled. Timings describe this host and workload.
"""
import argparse
import json
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import time
import urllib.error

import psutil
from validate_operation import Monitor, request
from validate_pretrained import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cache-mib', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--policies', type=int, nargs='+', default=[0, 25, 50, 100, -1])
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.cache_mib, args.threads) <= 0:
        parser.error('counts and budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'MODEL_ACCEPTANCE_MANIFEST.json').read_text())
    model_hash = sha256(args.model)
    model = next((m for m in manifest['models'] if m['publisher_sha256'] == model_hash
                  and m['size_bytes'] == args.model.stat().st_size), None)
    if model is None:
        parser.error('model does not match a pinned artifact')
    policies = args.policies
    if any(p < -1 or p > 100 for p in policies) or args.gpu_layers < 0:
        parser.error('invalid policy or GPU layer count')
    random.Random(1234).shuffle(policies)
    report = {'status': 'running', 'model': model, 'model_sha256': model_hash,
              'server_sha256': sha256(args.server), 'platform': platform.platform(),
              'system_ram_bytes': psutil.virtual_memory().total,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'policy_order': policies, 'warmup_requests': 1,
              'scope': 'Warm expert cache, no prompt KV reuse; placement and kernel mode in settings; one short prompt; not P100 acceptance',
              'runs': []}
    report['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit',
                                   '--format=csv'], capture_output=True, text=True).stdout.strip()
    body = {'prompt': 'Explain how a GPU works.', 'n_predict': args.tokens, 'seed': 1234,
            'temperature': 0, 'cache_prompt': False, 'return_tokens': True,
            'ignore_eos': True, 'stream': False}
    report['request'] = body

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        for policy in policies:
            label = 'adaptive' if policy == -1 else f'cpu-{policy}'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = [str(args.server.resolve()), '-m', str(args.model.resolve()),
                       '-ngl', str(args.gpu_layers), '-np', '1',
                       '-c', str(max(256, args.tokens + 128)), '-b', '128', '-ub', '8',
                       '-t', str(args.threads), '-tb', str(args.threads), '-fa', 'off',
                       '--load-mode', 'mmap', '--no-warmup', '--no-repack', '--fit', 'off',
                       '--moe-gpu-cache-mib', str(args.cache_mib), '--moe-device', 'CUDA0',
                       '--moe-staging-mib', '1', '--moe-compute-mib', '64',
                       '--moe-gpu-reserve-mib', '512', '--moe-cpu-miss-percent', str(policy),
                       '--host', '127.0.0.1', '--port', str(port)]
            if not args.gpu_layers:
                command.extend(['--no-op-offload', '--no-kv-offload'])
            if args.pipeline:
                command.append('--moe-pipeline')
            if args.fast:
                command.append('--moe-fast')
            run = {'policy': policy, 'label': label, 'command': command, 'requests': []}
            report['runs'].append(run)
            save()
            print(f'{label}: loading', flush=True)
            with (args.output / f'{label}.log').open('w', encoding='utf-8') as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                monitor = Monitor(proc.pid)
                try:
                    deadline = time.monotonic() + 900
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f'{label} exited during load: {proc.returncode}')
                        try:
                            if request(port, '/health', timeout=2).get('status') == 'ok':
                                break
                        except (urllib.error.URLError, TimeoutError):
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError('server readiness')
                        time.sleep(1)
                    for index in range(args.repeats + 1):
                        start = time.monotonic()
                        response = request(port, '/completion', body)
                        elapsed = time.monotonic() - start
                        if len(response.get('tokens', [])) != args.tokens:
                            raise RuntimeError('unexpected generated token count')
                        timing = response['timings']
                        if timing['predicted_per_second'] <= 0 or timing['prompt_n'] <= 0:
                            raise RuntimeError('invalid timing or reused prompt')
                        run['requests'].append({'warmup': index == 0, 'wall_seconds': elapsed,
                                                'response': response})
                        save()
                        print(f'{label} {"warmup" if index == 0 else index}: '
                              f'{timing["predicted_per_second"]:.3f} decode t/s, '
                              f'{timing["prompt_per_second"]:.3f} prompt t/s', flush=True)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=30)
                    run['memory'] = monitor.stop()
                    save()
        baseline_tokens = report['runs'][0]['requests'][0]['response']['tokens']
        report['identical_tokens'] = all(r['response']['tokens'] == baseline_tokens
                                        for run in report['runs'] for r in run['requests'])
        summary = []
        for run in report['runs']:
            measured = [r for r in run['requests'] if not r['warmup']]
            rates = [r['response']['timings']['predicted_per_second'] for r in measured]
            summary.append({'policy': run['policy'], 'label': run['label'],
                            'decode_median_tps': statistics.median(rates),
                            'decode_min_tps': min(rates), 'decode_max_tps': max(rates),
                            'prompt_median_tps': statistics.median(r['response']['timings']['prompt_per_second'] for r in measured),
                            'wall_median_seconds': statistics.median(r['wall_seconds'] for r in measured),
                            'peak_rss_gib': run['memory']['peak_rss'] / 2**30})
        baseline = next(s['decode_median_tps'] for s in summary if s['policy'] == 0)
        for row in summary:
            row['decode_change_percent'] = 100 * (row['decode_median_tps'] / baseline - 1)
        report['summary'] = sorted(summary, key=lambda s: s['decode_median_tps'], reverse=True)
        report['status'] = 'passed' if report['identical_tokens'] else 'token-divergence'
        print(json.dumps(report['summary'], indent=2), flush=True)
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        save()
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
