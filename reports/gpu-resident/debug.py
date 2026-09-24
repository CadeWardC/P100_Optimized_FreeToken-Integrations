from pathlib import Path
p=Path('llama.cpp/tests/test-moe-offload.cpp');s=p.read_text().replace('            device_close(actual, expected.output);','            std::fprintf(stderr, "boundary fast=%d step=%d actual=%g expected=%g\\n", int(fast), step, actual[0], expected.output[0]);\n            device_close(actual, expected.output);');p.write_text(s)
