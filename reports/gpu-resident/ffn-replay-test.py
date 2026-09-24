from pathlib import Path
p=Path('llama.cpp/tests/test-moe-offload.cpp');s=p.read_text().replace('    check(warm.workspace_builds == 1 && warm.workspace_reuses == 11, "decode rebuilt persistent workspace");','''    check(warm.workspace_builds == 1 && warm.workspace_reuses == 11, "decode rebuilt persistent workspace");
    std::printf("Persistent FFN CUDA captures=%llu replays=%llu\\n",
            (unsigned long long) warm.graph_captures, (unsigned long long) warm.graph_replays);
    if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
        check(warm.graph_captures > 0 && warm.graph_replays > 0, "persistent expert FFN did not replay");
    }''');p.write_text(s)
