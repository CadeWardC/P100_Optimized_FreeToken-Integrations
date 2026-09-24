<script lang="ts">
  import { Check, Copy, Sparkles, User } from 'lucide-svelte';
  import { marked } from 'marked';
  import { appState, type Message } from '../../stores/appState.svelte';
  import ReasoningBlock from './ReasoningBlock.svelte';

  let { msg } = $props<{ msg: Message }>();
  let copied = $state(false);

  let renderedContent = $derived(marked.parse(msg.content || ''));

  function copyText() {
    navigator.clipboard.writeText(msg.content);
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }
</script>

{#if msg.role === 'user'}
  <div class="flex justify-end my-4">
    <div class="flex gap-3 max-w-[85%]">
      <div class="bg-gradient-to-b from-[#1b253b] to-[#141b2c] border border-blue-500/20 rounded-2xl rounded-tr-sm px-4 py-3 text-slate-100 text-[14px] leading-relaxed shadow-lg">
        <div class="whitespace-pre-wrap select-text">{msg.content}</div>
      </div>
      <div class="w-8 h-8 rounded-full bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-300 shrink-0 shadow-inner">
        <User size={15} />
      </div>
    </div>
  </div>
{:else}
  <div class="flex justify-start my-4 group">
    <div class="flex gap-3 max-w-[92%] w-full">
      <div class="w-8 h-8 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0 shadow-inner">
        <Sparkles size={15} />
      </div>

      <div class="flex-1 bg-[#101622] border border-white/5 rounded-2xl rounded-tl-sm px-5 py-4 text-slate-200 text-[14px] leading-relaxed shadow-xl">
        {#if msg.reasoning}
          <ReasoningBlock reasoning={msg.reasoning} />
        {/if}

        <div class="prose prose-invert prose-emerald max-w-none prose-p:my-2 prose-pre:my-2 prose-headings:text-slate-100 select-text">
          {@html renderedContent}
        </div>

        <div class="flex items-center justify-between pt-3 mt-3 border-t border-white/5 text-[11px] text-slate-400 select-none">
          <div class="flex items-center gap-2 font-mono">
            {#if msg.stats}
              {#if msg.stats.tokSec}
                <span class="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {msg.stats.tokSec.toFixed(1)} t/s
                </span>
              {/if}
              {#if msg.stats.ttftMs}
                <span class="px-1.5 py-0.5 rounded bg-white/5 text-slate-300">
                  TTFT {Math.round(msg.stats.ttftMs)}ms
                </span>
              {/if}
              {#if appState.settings.show_output_tokens !== '0' && msg.stats.genTokens !== undefined}
                <span class="px-1.5 py-0.5 rounded bg-white/5 text-slate-300">
                  {msg.stats.genTokens} tokens
                </span>
              {/if}
              {#if msg.stats.stopReason}
                <span class="text-slate-400">({msg.stats.stopReason})</span>
              {/if}
            {/if}
          </div>

          <button
            onclick={copyText}
            class="flex items-center gap-1 px-2 py-1 rounded hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
            title="Copy response"
          >
            {#if copied}
              <Check size={12} class="text-emerald-400" />
              <span class="text-emerald-400 font-sans">Copied</span>
            {:else}
              <Copy size={12} />
              <span class="font-sans">Copy</span>
            {/if}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}
