<script lang="ts">
  import { Brain, ChevronDown, ChevronRight } from 'lucide-svelte';
  import { marked } from 'marked';

  let { reasoning = '' } = $props<{ reasoning?: string }>();
  let open = $state(false);

  let rendered = $derived(marked.parse(reasoning || ''));
</script>

{#if reasoning && reasoning.trim()}
  <div class="my-2 rounded-xl border border-purple-500/20 bg-[#120f1d]/60 overflow-hidden text-xs">
    <button
      onclick={() => open = !open}
      class="w-full flex items-center justify-between px-3.5 py-2 text-purple-300/80 hover:text-purple-200 hover:bg-purple-500/10 transition-colors text-left select-none"
    >
      <div class="flex items-center gap-2 font-medium">
        <Brain size={14} class="text-purple-400 animate-pulse" />
        <span>Thinking process</span>
      </div>
      {#if open}
        <ChevronDown size={14} />
      {:else}
        <ChevronRight size={14} />
      {/if}
    </button>

    {#if open}
      <div class="px-4 py-3 border-t border-purple-500/10 text-slate-300/80 leading-relaxed font-sans prose prose-invert prose-xs max-w-none border-l-2 border-l-purple-500/40 ml-2 mb-2">
        {@html rendered}
      </div>
    {/if}
  </div>
{/if}
