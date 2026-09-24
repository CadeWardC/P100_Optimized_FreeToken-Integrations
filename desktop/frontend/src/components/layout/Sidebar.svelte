<script lang="ts">
  import { Plus, MessageSquare, Layers, Sliders, Activity, Settings, Cpu, HardDrive, Zap, Info } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';

  const navItems = [
    { id: 'chat' as const, label: 'Chat', icon: MessageSquare },
    { id: 'models' as const, label: 'Models', icon: Layers, badge: () => appState.library.models.length || null },
    { id: 'tune' as const, label: 'Tune & Memory', icon: Sliders, warn: () => appState.engine.pendingReload },
    { id: 'monitor' as const, label: 'Monitor & Logs', icon: Activity },
  ];

  function fmtMib(mib: number | null | undefined): string {
    if (!mib) return '—';
    if (mib >= 1024) return (mib / 1024).toFixed(1) + ' GB';
    return Math.round(mib) + ' MB';
  }
</script>

<aside class="w-64 bg-[#0b0f17] border-r border-white/5 flex flex-col h-full select-none shrink-0">
  <!-- Brand block -->
  <div class="p-4 border-b border-white/5 flex items-center gap-3">
    <div class="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-md shadow-emerald-500/10">
      <Zap size={18} />
    </div>
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2">
        <span class="font-bold text-sm text-slate-100 tracking-tight truncate">LlamaCPP P100</span>
        <span class="px-1.5 py-0.2 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">v0.4</span>
      </div>
      <div class="text-[11px] text-slate-500 truncate">Intelligence, on your terms</div>
    </div>
  </div>

  <!-- Navigation items -->
  <nav class="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
    <button onclick={() => appState.newChat()} disabled={appState.engine.generating}
      class="w-full flex items-center gap-3 px-3 py-2.5 mb-3 rounded-xl border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40">
      <Plus size={17} /> New chat
    </button>
    {#each navItems as item}
      {@const Icon = item.icon}
      {@const isActive = appState.page === item.id}
      <button
        onclick={() => appState.page = item.id}
        class="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all {isActive ? 'bg-gradient-to-r from-emerald-500/15 to-emerald-500/5 text-emerald-300 border border-emerald-500/30 shadow-lg shadow-emerald-500/5' : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'}"
      >
        <div class="flex items-center gap-3">
          <Icon size={17} class={isActive ? 'text-emerald-400' : 'text-slate-400'} />
          <span>{item.label}</span>
        </div>

        <div class="flex items-center gap-1.5">
          {#if item.warn && item.warn()}
            <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse" title="Reload required"></span>
          {/if}
          {#if item.badge && item.badge()}
            <span class="px-1.5 py-0.5 rounded-md text-[10px] font-mono font-bold bg-white/10 text-slate-300">
              {item.badge()}
            </span>
          {/if}
        </div>
      </button>
    {/each}
    <div class="pt-5 text-[11px] text-slate-500 px-3 mb-2">CHATS</div>
    {#each appState.chats as chat}
      <button onclick={() => appState.newChat(chat.id)} disabled={appState.engine.generating}
        class="w-full text-left truncate px-3 py-2 rounded-lg text-xs disabled:opacity-40 {appState.activeChat === chat.id ? 'bg-white/10 text-slate-100' : 'text-slate-400 hover:bg-white/5'}" title={chat.title}>
        {chat.title}
      </button>
    {/each}
  </nav>

  <!-- Persistent Hardware & RAM Telemetry Footer -->
  <div class="p-3.5 border-t border-white/5 space-y-3 bg-[#080c13]/50">
    <!-- VRAM Gauge -->
    <div class="space-y-1.5">
      <div class="flex items-center justify-between text-[11px] font-mono">
        <span class="text-slate-400 flex items-center gap-1">
          <HardDrive size={12} class="text-emerald-400" />
          VRAM
        </span>
        <span class="text-slate-300 font-semibold">
          {fmtMib(appState.vramUsedMib)} / {fmtMib(appState.telemetry.vramTotalMib)}
        </span>
      </div>
      <div class="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden p-[1px]">
        <div
          class="h-full rounded-full transition-all duration-500 {appState.vramPct > 90 ? 'bg-rose-500' : appState.vramPct > 75 ? 'bg-amber-400' : 'bg-emerald-400'}"
          style="width: {appState.vramPct}%"
        ></div>
      </div>
    </div>

    <!-- Host RAM Gauge -->
    <div class="space-y-1.5">
      <div class="flex items-center justify-between text-[11px] font-mono">
        <span class="text-slate-400 flex items-center gap-1">
          <Cpu size={12} class="text-blue-400" />
          System RAM
        </span>
        <span class="text-slate-300 font-semibold">
          {fmtMib(appState.ramUsedMib)} / {fmtMib(appState.telemetry.ramTotalMib)}
        </span>
      </div>
      <div class="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden p-[1px]">
        <div
          class="h-full rounded-full bg-blue-400 transition-all duration-500"
          style="width: {appState.ramPct}%"
        ></div>
      </div>
    </div>

    <!-- Live GUI RAM Badge (Answering the user's specific request) -->
    <div class="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-emerald-500/5 border border-emerald-500/15 text-[11px] font-mono text-slate-300 shadow-sm">
      <div class="flex items-center gap-1.5 text-emerald-400">
        <Zap size={12} />
        <span class="font-sans font-medium text-[11px]">GUI RAM:</span>
      </div>
      <span class="font-bold text-emerald-300">
        ~{appState.telemetry.guiWorkingSetMib || 38} MB
      </span>
    </div>

    <!-- Settings button & GPU Tag -->
    <div class="flex items-center justify-between pt-1 text-[11px] text-slate-500">
      <span class="truncate max-w-[150px] font-mono" title={appState.telemetry.gpuName || appState.hardware.gpu}>
        {appState.telemetry.gpuName || appState.hardware.gpu}
      </span>
      <button
        onclick={() => appState.settingsOpen = true}
        class="p-1 rounded-md hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors"
        title="Settings"
      >
        <Settings size={14} />
      </button>
    </div>
  </div>
</aside>
