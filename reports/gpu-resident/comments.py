from pathlib import Path
p=Path('llama.cpp/src/llama-moe-offload.h');s=p.read_text().replace('// Serialized eager executor.', '// Serialized executor.');p.write_text(s)
