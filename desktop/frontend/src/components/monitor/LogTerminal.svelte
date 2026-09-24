<script lang="ts">
  import { Terminal, Search, Copy, Check, Pause, Play, ArrowDown, Trash2 } from 'lucide-svelte';
  import { appState, type LogEntry } from '../../stores/appState.svelte';

  let filter = $state<'all' | 'info' | 'warn' | 'err'>('all');
  let search = $state('');
  let autoscroll = $state(true);
  let paused = $state(false);
  let copied = $state(false);
  let terminalEl = $state<HTMLDivElement | null>(null);

  let filteredLogs = $derived.by(() => {
    let list = appState.logs;
    if (filter !== 'all') {
      list = list.filter(l => l.level === filter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(l => l.text.toLowerCase().includes(q));
    }
    return list;
  });

  $effect(() => {
    if (autoscroll && !paused && filteredLogs.length) {
      terminalEl?.scrollTo({ top: terminalEl.scrollHeight, behavior: 'smooth' });
    }
  });

  function copyAll() {
    const text = filteredLogs.map(l => l.text).join('\n');
    navigator.clipboard.writeText(text);
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }

  function clearLogs() {
    appState.logs = [];
  }
</script>

<div class="rounded-2xl bg-[#0c1017] border border-white/5 overflow-hidden flex flex-col h-full shadow-2xl">
  <!-- Terminal Top Toolbar -->
  <div class="p-3 bg-[#131824] border-b border-white/5 flex flex-wrap items-center justify-between gap-3 text-xs">
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 font-mono font-bold text-slate-200">
        <Terminal size={15} class="text-emerald-400" />
        <span>Live Engine Logs</span>
        <span class="px-1.5 py-0.2 rounded bg-white/5 text-[10px] font-mono text-slate-400">
          {filteredLogs.length} lines
        </span>
      </div>

      <!-- Filter tabs -->
      <div class="flex items-center gap-1 bg-[#090d14] p-1 rounded-xl border border-white/5">
        {#each ['all', 'info', 'warn', 'err'] as f}
          <button
            onclick={() => filter = f as any}
            class="px-2.5 py-0.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider transition-colors {filter === f ? 'bg-white/10 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}"
          >
            {f}
          </button>
        {/each}
      </div>
    </div>

    <div class="flex items-center gap-2">
      <!-- Search -->
      <div class="relative w-48">
        <Search size={13} class="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          bind:value={search}
          placeholder="Filter logs..."
          class="w-full pl-8 pr-2.5 py-1 bg-[#182133] border border-white/10 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
        />
      </div>

      <!-- Autoscroll Toggle -->
      <button
        onclick={() => autoscroll = !autoscroll}
        class="flex items-center gap-1 px-2.5 py-1 rounded-lg border text-[11px] transition-colors {autoscroll ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 font-semibold' : 'bg-white/5 text-slate-400 border-white/5'}"
        title="Toggle autoscroll"
      >
        <ArrowDown size={12} />
        <span>Autoscroll</span>
      </button>

      <!-- Copy -->
      <button
        onclick={copyAll}
        class="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5 text-[11px] transition-colors"
        title="Copy logs"
      >
        {#if copied}
          <Check size={12} class="text-emerald-400" />
          <span class="text-emerald-400">Copied</span>
        {:else}
          <Copy size={12} />
          <span>Copy</span>
        {/if}
      </button>

      <!-- Clear -->
      <button
        onclick={clearLogs}
        class="p-1 rounded-lg hover:bg-rose-500/10 hover:text-rose-400 text-slate-500 transition-colors"
        title="Clear terminal buffer"
      >
        <Trash2 size={13} />
      </button>
    </div>
  </div>

  <!-- Terminal Output Lines -->
  <div
    bind:this={terminalEl}
    class="flex-1 p-4 overflow-y-auto font-mono text-[12px] leading-relaxed select-text space-y-0.5 bg-[#080c13]"
  >
    {#if filteredLogs.length === 0}
      <div class="h-32 flex items-center justify-center text-slate-600 text-xs">
        No log output yet. Launch an engine or start inference to see streaming logs.
      </div>
    {:else}
      {#each filteredLogs as log (log.seq)}
        <div class="flex items-start gap-2 hover:bg-white/5 py-0.5 px-1.5 rounded transition-colors {log.level === 'err' ? 'text-rose-300 bg-rose-500/5' : log.level === 'warn' ? 'text-amber-300' : 'text-slate-300'}">
          <span class="text-slate-600 text-[10px] select-none shrink-0 w-12 font-mono">#{log.seq}</span>
          <span class="px-1 py-0.1 rounded text-[9px] font-bold uppercase shrink-0 select-none {log.level === 'err' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : log.level === 'warn' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' : 'bg-white/5 text-slate-500'}">
            {log.level}
          </span>
          <span class="break-all whitespace-pre-wrap flex-1">{log.text}</span>
        </div>
      {/each}
    {/if}
  </div>
</div>
