"""Compare edited chat continuation against an erased slot on a local GGUF server."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error

from validate_operation import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(args.server.resolve()), '-m', str(args.model.resolve()),
               '-ngl', '0', '--no-op-offload', '--no-kv-offload', '-np', '1',
               '-c', '512', '-b', '128', '-ub', '8', '-t', '8', '-tb', '8',
               '--no-warmup', '--no-repack', '--fit', 'off', '--moe-gpu-cache-mib', '256',
               '--moe-pipeline', '--semantic-checkpoints', '--checkpoint-min-step', '0',
               '--semantic-checkpoint-mib', '128', '--slots', '--cache-ram', '0',
               '--slot-save-path', str(args.output.resolve()),
               '--host', '127.0.0.1', '--port', str(port)]
    report = {'command': command, 'requests': [], 'status': 'running'}
    log_path = args.output / 'server.log'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited: {proc.returncode}')
                try:
                    if request(port, '/health', timeout=2).get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('server startup')
                time.sleep(1)
            messages = [
                {'role': 'user', 'content': 'Name two primary colors. Answer briefly.'},
                {'role': 'assistant', 'content': 'Red and blue.'},
                {'role': 'user', 'content': 'Now name one more color.'},
            ]
            def chat(label, content):
                body = {'messages': content, 'temperature': 0, 'seed': 1234,
                        'max_tokens': 4, 'cache_prompt': True, 'stream': False}
                response = request(port, '/v1/chat/completions', body)
                report['requests'].append({'label': label, 'request': body, 'response': response})
                return response['choices'][0]['message']
            chat('warm', messages)
            messages[-1]['content'] = 'Now name two more colors, briefly.'
            resumed = chat('edited', messages)
            request(port, '/slots/0?action=erase', {})
            fresh = chat('fresh', messages)
            if resumed != fresh:
                raise AssertionError('checkpoint continuation differs from fresh evaluation')
            log.flush()
            evidence = log_path.read_text(encoding='utf-8', errors='replace')
            report['saved'] = evidence.count('semantic checkpoint saved')
            report['restored'] = evidence.count('semantic checkpoint restored')
            if not report['saved'] or not report['restored']:
                raise AssertionError('semantic checkpoint save/restore not exercised')
            report['status'] = 'passed'
        except Exception as error:
            report['status'] = 'failed'
            report['error'] = str(error)
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
