from pathlib import Path
p=Path('llama.cpp/tests/test-moe-offload.cpp');s=p.read_text().replace('#include <cstdio>', '#include <cstdio>\n#include <cstdlib>');s=s.replace('        check(executor.metrics().evictions > 0, "device boundary did not evict");','''        check(executor.metrics().evictions > 0, "device boundary did not evict");
        const auto graphs = executor.metrics();
        std::printf("GPU boundary fast=%d CUDA captures=%llu replays=%llu\\n", int(fast),
                (unsigned long long) graphs.graph_captures, (unsigned long long) graphs.graph_replays);
        if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
            check(graphs.graph_captures > 0 && graphs.graph_replays > 0, "CUDA graph replay not exercised");
        }''');p.write_text(s)
p=Path('llama.cpp/tests/test-moe-integration.cpp');s=p.read_text();pos=s.index('    std::printf("PASS %s: full-vocabulary logits');s=s[:pos]+'''    if (device_boundary) {
        const auto host = llama_moe_memory(a.get()), device = llama_moe_memory(b.get());
        check(device.output_readback_bytes < host.output_readback_bytes, "pretrained device boundary was not used");
        std::printf("Device boundary output readback bytes: host=%llu device=%llu; CUDA captures=%llu replays=%llu\\n",
                (unsigned long long) host.output_readback_bytes, (unsigned long long) device.output_readback_bytes,
                (unsigned long long) device.expert_graph_captures, (unsigned long long) device.expert_graph_replays);
        if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
            check(device.expert_graph_captures > 0 && device.expert_graph_replays > 0, "pretrained CUDA replay not exercised");
        }
    }
''' +s[pos:];p.write_text(s)
