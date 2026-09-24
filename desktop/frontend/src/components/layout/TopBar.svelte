<script lang="ts">
  import { ChevronDown, Play, Square, Settings, RefreshCw, Layers } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';

  let showDropdown = $state(false);

  function handleModelSelect(id: string) {
    showDropdown = false;
    appState.selectModel(id);
  }
</script>

<header class="h-14 border-b border-white/5 bg-[#0b0f17]/90 backdrop-blur-md px-4 flex items-center justify-between z-20 shrink-0 select-none">
  <!-- Left: Active Model Pill with Status Badge -->
  <div class="flex items-center gap-3">
    <!-- Dropdown Trigger -->
    <div class="relative">
      <button
        onclick={() => showDropdown = !showDropdown}
        class="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#141b29] border border-white/10 hover:border-white/20 text-slate-200 text-xs font-medium transition-all shadow-sm group"
      >
        <!-- Status dot -->
        {#if appState.engine.state === 'ready'}
          {#if appState.engine.generating}
            <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
          {:else}
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
          {/if}
        {:else if appState.engine.state === 'loading'}
          <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
        {:else}
          <span class="w-2 h-2 rounded-full bg-slate-500"></span>
        {/if}

        <span class="font-semibold text-slate-100 max-w-[220px] truncate">
          {appState.activeModel ? appState.activeModel.name : 'No model selected'}
        </span>

        <ChevronDown size={14} class="text-slate-400 group-hover:text-white transition-transform {showDropdown ? 'rotate-180' : ''}" />
      </button>

      <!-- Quick Model Switcher Dropdown -->
      {#if showDropdown}
        <div
          class="absolute left-0 mt-2 w-80 rounded-xl bg-[#121824] border border-white/10 shadow-2xl z-50 p-1.5 space-y-1 overflow-hidden animate-in fade-in slide-in-from-top-1"
        >
          <div class="px-2.5 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Available Models ({appState.library.models.length})
          </div>

          <div class="max-h-64 overflow-y-auto space-y-0.5">
            {#each appState.library.models as m}
              <button
                onclick={() => handleModelSelect(m.id)}
                class="w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-left text-xs transition-colors {m.id === appState.settings.model ? 'bg-emerald-500/15 text-emerald-300 font-semibold' : 'text-slate-300 hover:bg-white/5'}"
              >
                <div class="truncate mr-2">
                  <div class="truncate">{m.name}</div>
                  <div class="text-[10px] text-slate-500 font-mono">
                    {(m.sizeBytes / (1024*1024*1024)).toFixed(1)} GB · {m.architecture}
                  </div>
                </div>
                {#if m.id === appState.settings.model}
                  <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0"></span>
                {/if}
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Live Status Label -->
    <div class="hidden sm:flex items-center gap-2 text-xs font-mono">
      {#if appState.engine.state === 'ready'}
        {#if appState.engine.generating}
          <span class="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-bold">
            Generating {appState.stream.tokSec ? `· ${appState.stream.tokSec.toFixed(1)} t/s` : ''}
          </span>
        {:else}
          <span class="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Ready
          </span>
        {/if}
      {:else if appState.engine.state === 'loading'}
        <span class="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
          Loading {appState.engine.percent > 0 ? `${appState.engine.percent}%` : ''}
          {appState.engine.stage ? `· ${appState.engine.stage}` : ''}
        </span>
      {:else if appState.engine.state === 'error'}
        <span class="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20" title={appState.engine.detail}>
          Startup failed
        </span>
      {:else}
        <span class="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-white/5">
          Engine Stopped
        </span>
      {/if}

      {#if appState.engine.pendingReload}
        <span class="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[11px] animate-pulse font-sans">
          Reload to apply changes
        </span>
      {/if}
    </div>
  </div>

  <!-- Right: Quick Engine Actions -->
  <div class="flex items-center gap-2">
    {#if appState.engine.state === 'stopped' || appState.engine.state === 'error'}
      <button
        onclick={() => appState.loadEngine()}
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-xs transition-all shadow-md shadow-emerald-500/10"
        title="Start llama-server"
      >
        <Play size={13} fill="currentColor" />
        <span>{appState.engine.state === 'error' ? 'Retry Load' : 'Load Model'}</span>
      </button>
    {:else if appState.engine.state === 'ready' && appState.engine.pendingReload}
      <button
        onclick={() => appState.loadEngine()}
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-black font-semibold text-xs transition-all shadow-md animate-pulse"
        title="Reload engine with new settings"
      >
        <RefreshCw size={13} />
        <span>Reload</span>
      </button>
    {:else if appState.engine.state === 'ready'}
      <button
        onclick={() => appState.unloadEngine()}
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-rose-500/20 hover:text-rose-400 text-slate-400 text-xs transition-all border border-white/5"
        title="Stop llama-server"
      >
        <Square size={12} />
        <span>Unload</span>
      </button>
    {/if}
  </div>
</header>
{#if appState.engine.state === 'error'}
  <div role="alert" class="px-4 py-3 text-sm text-rose-300 bg-rose-500/10 border-b border-rose-500/20">
    {appState.engine.detail || 'The engine could not start. Open Monitor & Logs for details.'}
  </div>
{/if}
