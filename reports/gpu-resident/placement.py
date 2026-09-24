from pathlib import Path
p=Path('llama.cpp/src/llama-context.cpp');s=p.read_text().replace('if (moe_executor && moe_executor->supports_device(ubatch.n_tokens) && strcmp(name, "ffn_moe_cpu") == 0)', 'if (moe_executor && moe_executor->supports_device(ubatch.n_tokens) && strcmp(name, "ffn_moe_cpu") == 0 &&\n                il >= 0 && model.dev_layer(il) == moe_executor->device())');p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.cpp');s=p.read_text().replace('if (external && op.executor && op.executor->supports_device(input->ne[1]))', 'if (external && op.executor && op.executor->supports_device(input->ne[1]) &&\n                !ggml_backend_buffer_is_host(dst->buffer))');p.write_text(s)
p=Path('llama.cpp/tests/test-moe-offload.cpp');s=p.read_text().replace('            device_close(actual, expected.output);','''            device_close(actual, expected.output);
            for (size_t d = 0; d < actual.size(); ++d) {
                check(std::abs(actual[d] - expected.output[d]) <= 1e-6f, "device merge changed same-mode arithmetic");
            }''');p.write_text(s)
