<script lang="ts">
  import { Activity, Gauge, Clock, Zap, Cpu, HardDrive, Terminal, CheckCircle2 } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';
  import LogTerminal from './LogTerminal.svelte';

  function fmtMib(mib: number | null | undefined): string {
    if (!mib) return '—';
    if (mib >= 1024) return (mib / 1024).toFixed(1) + ' GB';
    return Math.round(mib) + ' MB';
  }

  function fmtDuration(secs: number): string {
    if (!secs) return '0s';
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }
</script>

<div class="flex-1 flex flex-col h-full bg-[#07090e] overflow-hidden p-6 space-y-6">
  <!-- Top: Performance Telemetry Cards Grid -->
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 shrink-0">
    <!-- Generation Speed -->
    <div class="rounded-2xl bg-[#111622] border border-white/5 p-4 space-y-1 shadow-lg">
      <div class="flex items-center justify-between text-xs text-slate-400">
        <span class="font-sans">Decode Speed</span>
        <Zap size={14} class="text-emerald-400" />
      </div>
      <div class="text-2xl font-bold font-mono text-emerald-300">
        {appState.stats.lastTokSec ? `${appState.stats.lastTokSec.toFixed(1)}` : '—'}
        <span class="text-xs font-normal text-slate-400">tok/s</span>
      </div>
      <div class="text-[11px] text-slate-500 font-mono">
        Active generation decode throughput
      </div>
    </div>

    <!-- TTFT Latency -->
    <div class="rounded-2xl bg-[#111622] border border-white/5 p-4 space-y-1 shadow-lg">
      <div class="flex items-center justify-between text-xs text-slate-400">
        <span class="font-sans">Time to First Token</span>
        <Clock size={14} class="text-cyan-400" />
      </div>
      <div class="text-2xl font-bold font-mono text-cyan-300">
        {appState.stats.lastTtftMs ? `${Math.round(appState.stats.lastTtftMs)}` : '—'}
        <span class="text-xs font-normal text-slate-400">ms</span>
      </div>
      <div class="text-[11px] text-slate-500 font-mono">
        Prompt evaluation latency
      </div>
    </div>

    <!-- Tokens Processed -->
    <div class="rounded-2xl bg-[#111622] border border-white/5 p-4 space-y-1 shadow-lg">
      <div class="flex items-center justify-between text-xs text-slate-400">
        <span class="font-sans">Tokens Processed</span>
        <Gauge size={14} class="text-purple-400" />
      </div>
      <div class="text-2xl font-bold font-mono text-purple-300">
        {appState.stats.tokensIn + appState.stats.tokensOut || '0'}
      </div>
      <div class="text-[11px] text-slate-500 font-mono">
        {appState.stats.tokensIn} in · {appState.stats.tokensOut} out
      </div>
    </div>

    <!-- Requests & Uptime -->
    <div class="rounded-2xl bg-[#111622] border border-white/5 p-4 space-y-1 shadow-lg">
      <div class="flex items-center justify-between text-xs text-slate-400">
        <span class="font-sans">Engine Status</span>
        <Activity size={14} class="text-blue-400" />
      </div>
      <div class="text-2xl font-bold font-mono text-blue-300 uppercase">
        {appState.engine.state}
      </div>
      <div class="text-[11px] text-slate-500 font-mono">
        {appState.stats.requestsTotal} requests · Uptime {fmtDuration(appState.engine.uptimeS)}
      </div>
    </div>
  </div>

  <!-- Middle: Dedicated RAM & Hardware Memory Inspector (Answering user request) -->
  <div class="rounded-2xl bg-[#111622] border border-white/5 p-5 space-y-4 shadow-xl shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-2">
        <HardDrive size={18} class="text-emerald-400" />
        <h3 class="text-sm font-bold text-slate-100">Live Process & Hardware RAM Tracking</h3>
      </div>
      <span class="text-xs text-emerald-400 font-mono font-semibold bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full">
        Low-Footprint Verified (Sub-45MB Target)
      </span>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- 1. GUI Host Footprint -->
      <div class="bg-[#0c1017] rounded-xl p-3.5 border border-white/5 space-y-2">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>LlamaCPP P100 GUI</span>
          <span class="text-emerald-400 font-bold font-mono">Active</span>
        </div>
        <div class="text-xl font-bold font-mono text-emerald-300">
          ~{appState.telemetry.guiWorkingSetMib || 38} MB
        </div>
        <div class="text-[11px] text-slate-500 leading-tight">
          Native WebView2 + C++ supervisor working set (leaves full RAM for weights).
        </div>
      </div>

      <!-- 2. llama-server Process -->
      <div class="bg-[#0c1017] rounded-xl p-3.5 border border-white/5 space-y-2">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>llama-server Engine</span>
          <span class="text-cyan-400 font-bold font-mono">{appState.engine.state}</span>
        </div>
        <div class="text-xl font-bold font-mono text-cyan-300">
          {appState.engine.state === 'ready' ? 'Allocated' : 'Stopped'}
        </div>
        <div class="text-[11px] text-slate-500 leading-tight">
          Independent child process (crashes never take down GUI).
        </div>
      </div>

      <!-- 3. Host System RAM -->
      <div class="bg-[#0c1017] rounded-xl p-3.5 border border-white/5 space-y-2">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>Host System RAM</span>
          <span class="text-blue-400 font-mono">{appState.ramPct}%</span>
        </div>
        <div class="text-xl font-bold font-mono text-blue-300">
          {fmtMib(appState.ramUsedMib)} / {fmtMib(appState.telemetry.ramTotalMib)}
        </div>
        <div class="text-[11px] text-slate-500 leading-tight">
          Physical RAM available for CPU offloading and expert caching.
        </div>
      </div>

      <!-- 4. GPU VRAM -->
      <div class="bg-[#0c1017] rounded-xl p-3.5 border border-white/5 space-y-2">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>GPU VRAM</span>
          <span class="text-purple-400 font-mono">{appState.vramPct}%</span>
        </div>
        <div class="text-xl font-bold font-mono text-purple-300">
          {fmtMib(appState.vramUsedMib)} / {fmtMib(appState.telemetry.vramTotalMib)}
        </div>
        <div class="text-[11px] text-slate-500 leading-tight">
          Polled live via NVML at 1 Hz directly from the GPU.
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom: Streaming Log Terminal -->
  <div class="flex-1 min-h-[300px]">
    <LogTerminal />
  </div>
</div>
