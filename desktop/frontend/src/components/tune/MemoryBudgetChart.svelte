<script lang="ts">
  import { HardDrive, Cpu, AlertTriangle, Info } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';

  function fmtMib(n: number) {
    if (n >= 1024) return (n / 1024).toFixed(1) + ' GB';
    return Math.round(n) + ' MB';
  }

  let vramTotal = $derived(appState.telemetry.vramTotalMib || 16384);
  let ramTotal = $derived(appState.telemetry.ramTotalMib || 16384);

  let vramSegments = $derived(appState.review.budget.filter(b => b.device === 'vram'));
  let ramSegments = $derived(appState.review.budget.filter(b => b.device === 'ram'));

  let vramUsedTotal = $derived(vramSegments.reduce((acc, b) => acc + b.mib, 0));
  let ramUsedTotal = $derived(ramSegments.reduce((acc, b) => acc + b.mib, 0));

  function getColor(kind: string): string {
    switch (kind) {
      case 'weights': return 'bg-emerald-400';
      case 'kv': return 'bg-cyan-400';
      case 'cache': return 'bg-purple-400';
      case 'reserve': return 'bg-amber-400';
      default: return 'bg-blue-400';
    }
  }

  function getBadgeColor(kind: string): string {
    switch (kind) {
      case 'weights': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
      case 'kv': return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20';
      case 'cache': return 'text-purple-400 bg-purple-500/10 border-purple-500/20';
      case 'reserve': return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
      default: return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
    }
  }
</script>

<div class="rounded-2xl bg-[#111622] border border-white/5 p-6 space-y-6 shadow-xl">
  <div>
    <h3 class="text-base font-bold text-slate-100 flex items-center gap-2">
      <HardDrive size={18} class="text-emerald-400" />
      Estimated Memory Budget Plan
    </h3>
    <p class="text-xs text-slate-400 mt-1">
      Theoretical memory breakdown calculated from the GGUF model dimensions and current profile settings.
    </p>
  </div>

  <!-- VRAM Stacked Bar -->
  <div class="space-y-2">
    <div class="flex items-center justify-between text-xs font-mono">
      <span class="text-slate-300 font-semibold flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
        GPU VRAM Allocation
      </span>
      <span class="text-slate-400">
        <span class="text-emerald-300 font-bold">{fmtMib(vramUsedTotal)}</span> / {fmtMib(vramTotal)}
        ({Math.min(100, Math.round((vramUsedTotal / vramTotal) * 100))}%)
      </span>
    </div>

    <div class="h-4 w-full bg-[#0a0d14] rounded-xl overflow-hidden p-0.5 border border-white/5 flex gap-0.5">
      {#each vramSegments as seg}
        {@const pct = Math.max(1, (seg.mib / vramTotal) * 100)}
        <div
          class="h-full rounded-sm {getColor(seg.kind)} transition-all duration-300 relative group cursor-pointer"
          style="width: {pct}%"
          title="{seg.label}: {fmtMib(seg.mib)}"
        ></div>
      {/each}
      <div class="h-full flex-1 bg-white/5 rounded-sm" title="Free Headroom"></div>
    </div>
  </div>

  <!-- System RAM Stacked Bar -->
  <div class="space-y-2">
    <div class="flex items-center justify-between text-xs font-mono">
      <span class="text-slate-300 font-semibold flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span>
        Host RAM Allocation (CPU Offload)
      </span>
      <span class="text-slate-400">
        <span class="text-blue-300 font-bold">{fmtMib(ramUsedTotal)}</span> / {fmtMib(ramTotal)}
        ({Math.min(100, Math.round((ramUsedTotal / ramTotal) * 100))}%)
      </span>
    </div>

    <div class="h-4 w-full bg-[#0a0d14] rounded-xl overflow-hidden p-0.5 border border-white/5 flex gap-0.5">
      {#each ramSegments as seg}
        {@const pct = Math.max(1, (seg.mib / ramTotal) * 100)}
        <div
          class="h-full rounded-sm {getColor(seg.kind)} transition-all duration-300 relative group cursor-pointer"
          style="width: {pct}%"
          title="{seg.label}: {fmtMib(seg.mib)}"
        ></div>
      {/each}
      <div class="h-full flex-1 bg-white/5 rounded-sm" title="Free Headroom"></div>
    </div>
  </div>

  <!-- Budget Breakdown Table -->
  <div class="overflow-x-auto">
    <table class="w-full text-left text-xs font-mono">
      <thead>
        <tr class="text-slate-500 border-b border-white/5 uppercase text-[10px]">
          <th class="py-2 px-3">Segment</th>
          <th class="py-2 px-3">Device</th>
          <th class="py-2 px-3">Size</th>
          <th class="py-2 px-3 font-sans">Details</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-white/5 text-slate-300">
        {#each appState.review.budget as seg}
          <tr class="hover:bg-white/5 transition-colors">
            <td class="py-2 px-3 font-semibold text-slate-100 flex items-center gap-2">
              <span class="w-2 h-2 rounded-full {getColor(seg.kind)}"></span>
              {seg.label}
            </td>
            <td class="py-2 px-3">
              <span class="px-2 py-0.5 rounded text-[10px] uppercase {seg.device === 'vram' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'}">
                {seg.device}
              </span>
            </td>
            <td class="py-2 px-3 font-bold">{fmtMib(seg.mib)}</td>
            <td class="py-2 px-3 font-sans text-slate-400 text-[11px]">{seg.note}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Warnings Callout Cards -->
  {#if appState.review.warnings.length > 0}
    <div class="space-y-2 pt-2 border-t border-white/5">
      {#each appState.review.warnings as w}
        <div class="flex items-start gap-2.5 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200">
          <AlertTriangle size={15} class="text-amber-400 shrink-0 mt-0.5" />
          <span class="leading-relaxed">{w}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>
