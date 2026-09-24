<script lang="ts">
  import { Check, Eye, Play, Square, AlertCircle, HardDrive, Layers, Cpu, Sparkles } from 'lucide-svelte';
  import { appState, type ModelItem } from '../../stores/appState.svelte';

  let { model } = $props<{ model: ModelItem }>();

  let isSelected = $derived(appState.settings.model === model.id);
  let isLoaded = $derived(isSelected && appState.engine.state === 'ready');
  let isLoading = $derived(isSelected && appState.engine.state === 'loading');

  let sizeGb = $derived((model.sizeBytes / (1024 * 1024 * 1024)).toFixed(1));
  let quantTag = $derived.by(() => {
    const match = model.name.match(/(Q[0-9]_[A-Z0-9_]+|F16|BF16)/i);
    return match ? match[1].toUpperCase() : 'GGUF';
  });

  // VRAM Fit Estimation
  let vramFit = $derived.by(() => {
    const vramMib = appState.telemetry.vramTotalMib || 16384;
    const modelMib = model.sizeBytes / (1024 * 1024);
    const estReq = modelMib * 1.15 + 1024; // weights + 15% KV + scratch
    if (estReq <= vramMib * 0.9) {
      return { status: 'fit', label: '100% GPU Fit', desc: `Fits easily in ${Math.round(vramMib/1024)}GB VRAM`, color: 'emerald' };
    } else if (modelMib < vramMib) {
      return { status: 'tight', label: 'Tight GPU Fit', desc: 'Fits with reduced context/scratch', color: 'amber' };
    } else {
      return { status: 'offload', label: 'Hybrid Offload', desc: 'GPU layers + CPU host RAM offload', color: 'blue' };
    }
  });

  async function handleLoadToggle() {
    if (isLoaded) {
      await appState.unloadEngine();
    } else {
      if (!isSelected) {
        await appState.selectModel(model.id);
      } else {
        await appState.loadEngine();
      }
    }
  }
</script>

<div class="rounded-2xl bg-[#111622] border transition-all duration-300 relative overflow-hidden flex flex-col justify-between shadow-xl group {isSelected ? 'border-emerald-500/40 ring-1 ring-emerald-500/20 shadow-emerald-500/5' : 'border-white/5 hover:border-white/15 hover:bg-[#141b2a]'}">
  <!-- Card Header -->
  <div class="p-5 space-y-3">
    <div class="flex items-start justify-between gap-2">
      <div class="space-y-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            {quantTag}
          </span>
          <span class="px-2 py-0.5 rounded-md text-[10px] font-mono bg-white/5 text-slate-300 border border-white/5">
            {model.architecture || 'Llama'}
          </span>
          {#if model.projector}
            <span class="px-1.5 py-0.5 rounded-md text-[10px] bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center gap-1" title="Vision Projector Paired">
              <Eye size={11} />
              <span>Vision</span>
            </span>
          {/if}
          {#if model.experts > 0}
            <span class="px-1.5 py-0.5 rounded-md text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              {model.experts}x MoE
            </span>
          {/if}
        </div>

        <h3 class="font-bold text-slate-100 text-sm tracking-tight truncate group-hover:text-white" title={model.name}>
          {model.name}
        </h3>
      </div>

      {#if isSelected}
        <div class="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0" title="Active Model">
          <Check size={14} />
        </div>
      {/if}
    </div>

    <!-- Specs Grid -->
    <div class="grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-400 pt-1">
      <div class="bg-black/20 rounded-lg p-2 border border-white/5">
        <div class="text-[10px] text-slate-500 uppercase">Size</div>
        <div class="font-semibold text-slate-200">{sizeGb} GB</div>
      </div>
      <div class="bg-black/20 rounded-lg p-2 border border-white/5">
        <div class="text-[10px] text-slate-500 uppercase">Context</div>
        <div class="font-semibold text-slate-200">{model.context ? `${Math.round(model.context/1024)}k` : '4k'}</div>
      </div>
      <div class="bg-black/20 rounded-lg p-2 border border-white/5">
        <div class="text-[10px] text-slate-500 uppercase">Layers</div>
        <div class="font-semibold text-slate-200">{model.layers || '—'}</div>
      </div>
    </div>

    <!-- VRAM Fit Meter -->
    <div class="pt-1">
      <div class="flex items-center justify-between text-[11px] mb-1">
        <span class="text-slate-400 font-sans">Hardware Fit:</span>
        <span class="font-semibold font-mono text-{vramFit.color}-400">{vramFit.label}</span>
      </div>
      <div class="text-[10px] text-slate-500">{vramFit.desc}</div>
    </div>

    <!-- Live Loading Progress Bar (if active) -->
    {#if isLoading}
      <div class="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 space-y-2 animate-pulse">
        <div class="flex items-center justify-between text-xs font-semibold text-amber-300">
          <span>{appState.engine.stage || 'Loading tensors...'}</span>
          <span>{appState.engine.percent > 0 ? `${appState.engine.percent}%` : ''}</span>
        </div>
        <div class="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
          <div
            class="h-full bg-amber-400 transition-all duration-300"
            style="width: {Math.max(5, appState.engine.percent)}%"
          ></div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Card Actions -->
  <div class="p-4 pt-0 flex items-center gap-2">
    <button
      onclick={() => appState.selectModel(model.id)}
      class="flex-1 py-2 px-3 rounded-xl text-xs font-semibold transition-all border {isSelected ? 'bg-white/5 text-slate-300 border-white/10' : 'bg-transparent hover:bg-white/5 text-slate-400 hover:text-white border-white/5'}"
    >
      {isSelected ? 'Active Model' : 'Select'}
    </button>

    <button
      onclick={handleLoadToggle}
      disabled={isLoading}
      class="py-2 px-4 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 shadow-md {isLoaded ? 'bg-rose-500/15 hover:bg-rose-500/25 text-rose-400 border border-rose-500/30' : 'bg-emerald-500 hover:bg-emerald-400 text-black shadow-emerald-500/10 disabled:opacity-50'}"
    >
      {#if isLoaded}
        <Square size={12} />
        <span>Unload</span>
      {:else}
        <Play size={12} fill="currentColor" />
        <span>Load</span>
      {/if}
    </button>
  </div>
</div>
