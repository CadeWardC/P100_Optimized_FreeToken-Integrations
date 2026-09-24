<script lang="ts">
  import { Search, RefreshCw, Layers } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';
  import ModelCard from './ModelCard.svelte';

  let searchQuery = $state('');
  let filteredModels = $derived.by(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return appState.library.models;
    return appState.library.models.filter(m =>
      m.name.toLowerCase().includes(q) ||
      m.architecture.toLowerCase().includes(q)
    );
  });
</script>

<div class="flex-1 flex flex-col h-full bg-[#07090e] overflow-hidden">
  <!-- Top Tool Bar -->
  <div class="p-6 border-b border-white/5 bg-[#0b0f17]/40 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
    <div>
      <h2 class="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2.5">
        <Layers size={20} class="text-emerald-400" />
        Model Library
      </h2>
      <p class="text-xs text-slate-400 mt-1">
        Discovered {appState.library.models.length} GGUF models in
        <code class="px-1.5 py-0.5 rounded bg-white/5 text-slate-300 font-mono text-[11px]">{appState.library.folder || './models'}</code>
      </p>
    </div>

    <div class="flex items-center gap-3">
      <!-- Search -->
      <div class="relative w-full sm:w-64">
        <Search size={14} class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          bind:value={searchQuery}
          placeholder="Filter models..."
          class="w-full pl-9 pr-3 py-1.5 bg-[#111724] border border-white/10 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
        />
      </div>

      <!-- Rescan Button -->
      <button
        onclick={() => appState.scanFolder()}
        disabled={appState.scanning}
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white border border-white/10 text-xs font-semibold transition-all disabled:opacity-50 shrink-0"
      >
        <RefreshCw size={13} class={appState.scanning ? 'animate-spin text-emerald-400' : ''} />
        <span>{appState.scanning ? 'Scanning...' : 'Rescan'}</span>
      </button>
    </div>
  </div>

  <!-- Cards Grid Container -->
  <div class="flex-1 overflow-y-auto p-6">
    {#if filteredModels.length === 0}
      <div class="h-64 flex flex-col items-center justify-center text-center">
        <Layers size={32} class="text-slate-600 mb-3" />
        <h3 class="text-sm font-semibold text-slate-300">No models found</h3>
        <p class="text-xs text-slate-500 max-w-sm mt-1">
          {searchQuery ? 'No models match your search query.' : 'Choose your model folder in Settings to find GGUF models.'}
        </p>
      </div>
    {:else}
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5 max-w-7xl mx-auto">
        {#each filteredModels as model (model.id)}
          <ModelCard {model} />
        {/each}
      </div>
    {/if}
  </div>
</div>

