<script lang="ts">
  import { Check, Copy } from 'lucide-svelte';
  import hljs from 'highlight.js';

  let { code = '', lang = 'text' } = $props<{ code?: string; lang?: string }>();
  let copied = $state(false);

  let highlighted = $derived.by(() => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value;
      } catch (_) {}
    }
    return hljs.highlightAuto(code).value;
  });

  function copyCode() {
    navigator.clipboard.writeText(code);
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }
</script>

<div class="my-3 rounded-xl overflow-hidden border border-white/10 bg-[#0c1017] text-slate-200 text-xs shadow-xl">
  <div class="flex items-center justify-between px-3.5 py-1.5 bg-[#151c28] border-b border-white/5 text-slate-400 font-mono text-[11px]">
    <span class="font-medium uppercase tracking-wider text-slate-400">{lang || 'code'}</span>
    <button
      onclick={copyCode}
      class="flex items-center gap-1.5 px-2 py-0.5 rounded hover:bg-white/10 text-slate-300 hover:text-white transition-colors"
      title="Copy code"
    >
      {#if copied}
        <Check size={13} class="text-emerald-400" />
        <span class="text-emerald-400 font-sans">Copied!</span>
      {:else}
        <Copy size={13} />
        <span class="font-sans">Copy</span>
      {/if}
    </button>
  </div>
  <pre class="p-3.5 overflow-x-auto text-[13px] leading-relaxed font-mono bg-[#0a0e16]"><code>{@html highlighted}</code></pre>
</div>
