"""Load a local K2-Horizon GGUF and smoke-test completion and chat."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    command = [str(args.server.resolve()), '-m', str(args.model.resolve()),
               '--host', '127.0.0.1', '--port', str(port), '-ngl', '99',
               '-c', '2048', '-b', '128', '-ub', '128', '-np', '1',
               '-fa', 'off', '--jinja', '--reasoning-budget', '0']
    report = {'model': str(args.model.resolve()), 'command': command}
    with (args.output / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited with {process.returncode}; see server.log')
                try:
                    report['health'] = request('/health')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Model did not become ready; see server.log')
                    time.sleep(0.25)
            report['completion'] = request('/completion', {
                'prompt': 'The capital of France is', 'n_predict': 32,
                'temperature': 0, 'seed': 42, 'cache_prompt': False,
            })
            if not report['completion'].get('content', '').strip():
                raise RuntimeError('Completion returned no text')
            report['chat'] = request('/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 128,
                'chat_template_kwargs': {'enable_thinking': False},
            })
            if not report['chat']['choices'][0]['message'].get('content', '').strip():
                raise RuntimeError('Chat returned no final answer')
            report['passed'] = True
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'],
                      'completion': report['completion']['content'],
                      'chat': report['chat']['choices'][0]['message']}, indent=2))


if __name__ == '__main__':
    main()
