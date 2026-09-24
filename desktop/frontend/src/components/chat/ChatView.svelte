<script lang="ts">
  import { Sparkles } from 'lucide-svelte';
  import { marked } from 'marked';
  import { appState } from '../../stores/appState.svelte';
  import MessageBubble from './MessageBubble.svelte';
  import ReasoningBlock from './ReasoningBlock.svelte';
  import Composer from './Composer.svelte';

  let messagesEndEl = $state<HTMLDivElement | null>(null);

  $effect(() => {
    if (appState.conversation.length || appState.stream.content) {
      messagesEndEl?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  const suggestions = [
    { title: 'Explain MoE expert caching', desc: 'How does FreeToken utilize GPU & CPU offload on P100?' },
    { title: 'Write a CUDA C++ kernel', desc: 'Implement high-throughput FP16 vector dot product in C++.' },
    { title: 'Compare GGUF quantizations', desc: 'Break down perplexity vs RAM tradeoffs for Q4_K_M vs Q8_0.' },
  ];

  let renderedStream = $derived(marked.parse(appState.stream.content || ''));
</script>

<div class="flex-1 flex flex-col h-full bg-[#07090e] overflow-hidden relative">
  <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-2">
    <div class="max-w-4xl mx-auto">
      {#if appState.conversation.length === 0 && !appState.stream.active}
        <div class="h-full flex flex-col items-center justify-center text-center py-16 px-4">
          <div class="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-6 shadow-xl shadow-emerald-500/5">
            <Sparkles size={32} class="text-emerald-400" />
          </div>

          <h2 class="text-2xl font-bold text-slate-100 tracking-tight mb-2">
            Intelligence, on your terms.
          </h2>
          <p class="text-sm text-slate-400 max-w-md mb-8 leading-relaxed">
            {#if appState.activeModel}
              Ready to chat with <span class="text-emerald-400 font-semibold">{appState.activeModel.name}</span>.
            {:else}
              Select a model from the Models tab to begin local inference on your GPU.
            {/if}
          </p>

          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-2xl">
            {#each suggestions as s}
              <button
                onclick={() => appState.sendMessage(s.desc)}
                class="flex flex-col text-left p-4 rounded-xl bg-[#111724] border border-white/5 hover:border-emerald-500/30 hover:bg-[#151d2d] transition-all group shadow-lg"
              >
                <span class="text-xs font-semibold text-slate-200 group-hover:text-emerald-400 mb-1">{s.title}</span>
                <span class="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">{s.desc}</span>
              </button>
            {/each}
          </div>
        </div>
      {:else}
        {#each appState.conversation as msg}
          <MessageBubble {msg} />
        {/each}

        {#if appState.stream.active}
          <div class="flex justify-start my-4">
            <div class="flex gap-3 max-w-[92%] w-full">
              <div class="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shrink-0 shadow-md">
                <Sparkles size={15} class="animate-pulse" />
              </div>

              <div class="flex-1 bg-[#101622] border border-emerald-500/20 rounded-2xl rounded-tl-sm px-5 py-4 text-slate-200 text-[14px] leading-relaxed shadow-2xl">
                {#if appState.stream.reasoning}
                  <ReasoningBlock reasoning={appState.stream.reasoning} />
                {/if}

                <div class="prose prose-invert prose-emerald max-w-none prose-p:my-2 prose-pre:my-2 select-text">
                  {@html renderedStream}
                  <span class="inline-block w-2 h-4 ml-1 bg-emerald-400 animate-pulse-glow rounded-sm translate-y-0.5"></span>
                </div>

                <div class="flex items-center gap-3 pt-3 mt-3 border-t border-white/5 text-[11px] text-slate-400 font-mono">
                  {#if appState.stream.tokSec}
                    <span class="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                      {appState.stream.tokSec.toFixed(1)} t/s
                    </span>
                  {/if}
                  {#if appState.settings.show_output_tokens !== '0'}<span>{appState.outputTokens} output tokens</span>{/if}
                  {#if appState.stream.ttftMs !== null}<span>TTFT {Math.round(appState.stream.ttftMs)}ms</span>{/if}
                  <span class="text-slate-500">{appState.stream.content || appState.stream.reasoning ? 'Generating response...' : 'Processing prompt...'}</span>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <div bind:this={messagesEndEl}></div>
      {/if}
    </div>
  </div>

  <Composer />
</div>
