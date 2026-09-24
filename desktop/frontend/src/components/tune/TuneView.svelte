<script lang="ts">
  import { appState } from '../../stores/appState.svelte';
  import MemoryBudgetChart from './MemoryBudgetChart.svelte';
  import SettingControl from './SettingControl.svelte';
  import { options, expertKeys, textKeys, inference } from './fields';
  let tab = $state('inference');
  let search = $state('');
  let baseline = $derived(appState.settings.freetoken_model === appState.settings.model ? appState.settings.freetoken_baseline || {} : {});
  let fields = $derived(tab === 'inference' ? inference : tab === 'freetoken' ? [
    {key:'expert_enabled',label:'FreeToken expert features enabled',initial:'1',toggle:true,help:'Turn off to bypass custom expert settings while keeping their values.'},
    ...appState.loadFields.filter(f => expertKeys.includes(f.key)),
    {key:'graphs',label:'CUDA graphs',initial:'auto',help:'Automatic backend graphs or off. Semantic checkpoint profiles use off.'}
  ] : [
    {key:'auto_settings',label:'Recommend settings when switching models',initial:'1',toggle:true},
    ...appState.loadFields.filter(f => !expertKeys.includes(f.key))
  ]);
  let visible = $derived(fields.map(f => ({...f, options:options[f.key] || f.options, text:textKeys.includes(f.key), multiline:f.multiline || f.key==='chat_template'})).filter(f => `${f.label} ${f.help || ''}`.toLowerCase().includes(search.toLowerCase())));
</script>
<div class="flex-1 flex flex-col min-h-0 bg-[#07090e]">
  <div class="flex-1 overflow-y-auto p-6">
    <div class="max-w-4xl mx-auto space-y-6">
      <div class="flex justify-between items-start gap-4">
        <div><h2 class="text-xl font-bold text-slate-100">Model settings</h2><p class="text-sm text-slate-400 mt-1">Tune responses, memory, and FreeToken for your machine.</p></div>
        <span class="text-xs text-slate-500 pt-2">{appState.savingSettings ? 'Saving…' : 'Changes save automatically'}</span>
      </div>
      <section class="rounded-2xl p-5 bg-emerald-500/5 border border-emerald-500/25 space-y-4">
        <div><h3 class="text-emerald-300 font-semibold">FreeToken automatic setup</h3><p class="text-xs text-slate-400 mt-1">Detect this machine, choose compatible features, and save a baseline you can restore after manual changes.</p></div>
        <div class="flex flex-wrap gap-3">
          <button onclick={() => appState.configureFreeToken('auto')} disabled={!appState.activeModel || appState.engine.generating || appState.engine.state==='loading'} class="primary">Enable & auto-configure FreeToken</button>
          <button onclick={() => appState.configureFreeToken('restore')} disabled={!Object.keys(baseline).length} class="secondary">Restore automatic settings</button>
          <button onclick={() => appState.configureFreeToken('disable')} class="secondary">Disable FreeToken</button>
        </div>
        {#if Object.keys(baseline).length}<p class="text-xs text-slate-400 leading-relaxed">{appState.settings.freetoken_note}</p>{/if}
      </section>
      <section class="rounded-2xl bg-[#111622] border border-white/10 p-5 space-y-4">
        <div class="flex justify-between items-center"><h3 class="font-semibold text-slate-100">Response length</h3><span class="text-xs font-mono text-emerald-400">Output: {appState.settings.max_tokens === '-1' ? 'No limit' : `${appState.settings.max_tokens || 512} tokens`}</span></div>
        <div class="flex flex-wrap items-center gap-4">
          <label class="text-sm text-slate-300 flex items-center gap-2"><input type="checkbox" checked={appState.settings.max_tokens === '-1'} onchange={e => appState.queueSaveSetting('max_tokens', e.currentTarget.checked ? '-1' : '512')} class="accent-emerald-400" />No output token limit</label>
          {#if appState.settings.max_tokens !== '-1'}<label class="text-sm text-slate-400 flex items-center gap-2">Maximum tokens <input aria-label="Maximum output tokens" type="number" min="1" max="1048576" step="1" value={appState.settings.max_tokens || '512'} onchange={e => appState.queueSaveSetting('max_tokens', e.currentTarget.value)} class="w-32 rounded-lg p-2 bg-[#090e17] border border-white/10 text-slate-200" /></label>{/if}
          <label class="text-sm text-slate-300 flex items-center gap-2"><input type="checkbox" checked={appState.settings.show_output_tokens !== '0'} onchange={e => appState.queueSaveSetting('show_output_tokens', e.currentTarget.checked ? '1' : '0')} class="accent-emerald-400" />Show output token count</label>
        </div>
        <p class="text-xs text-slate-500">No limit removes the response token cap. The model can still finish naturally, reach its context limit, match a stop string, or be stopped by you.</p>
      </section>
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div class="flex gap-2" role="tablist" aria-label="Setting groups">{#each [['inference','Inference'],['load','Model loading'],['freetoken','FreeToken']] as [id,label]}<button role="tab" aria-selected={tab===id} onclick={() => tab=id} class={tab===id ? 'primary' : 'secondary'}>{label}</button>{/each}</div>
        <input aria-label="Search settings" placeholder="Find a setting…" bind:value={search} class="rounded-xl bg-[#111622] border border-white/10 px-3 py-2 text-sm text-slate-200" />
      </div>
      {#if tab!=='inference'}<p class="text-xs text-amber-300">Loading and FreeToken changes take effect after loading or reloading the model.</p>{/if}
      <section class="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6 rounded-2xl bg-[#111622] border border-white/5 p-6">
        {#each visible as field (field.key)}<SettingControl {field} baseline={baseline[field.key]} />{/each}
        {#if !visible.length}<p class="text-sm text-slate-400">No matching settings.</p>{/if}
      </section>
      {#if tab==='freetoken'}<p class="text-xs text-slate-500">GPU and CPU caches are separate modes. Native numerical mode may change results and cannot be combined with semantic checkpoints. Automatic setup chooses a compatible combination; all controls remain editable.</p><MemoryBudgetChart />{/if}
      <details class="text-xs text-slate-500 border-t border-white/5 pt-4"><summary class="cursor-pointer">LM Studio settings compatibility</summary><p class="mt-3 leading-relaxed">These controls cover llama.cpp model loading and inference. Apple MLX and ONNX options, LM Studio’s GPU driver memory cap, legacy tail-free sampling, and its tool execution system are not available in this engine. Context overflow uses the engine’s context-shift or stop behavior. Model and backend support determine whether a cache type, template, or draft model can run.</p></details>
    </div>
  </div>
  {#if appState.engine.pendingReload}<div class="flex items-center justify-between gap-4 px-6 py-3 bg-amber-500/10 border-t border-amber-500/25 text-sm text-amber-300"><span>Saved settings need a model reload.</span><button class="primary" onclick={() => appState.loadEngine()} disabled={appState.engine.generating}>Reload model</button></div>{/if}
</div>
<style>
  .primary { background:#34d399; color:#07110d; border-radius:10px; padding:9px 14px; font-size:12px; font-weight:650; }
  .secondary { background:#ffffff08; color:#cbd5e1; border:1px solid #ffffff15; border-radius:10px; padding:9px 14px; font-size:12px; }
  button:disabled { opacity:.4; cursor:not-allowed; }
</style>
