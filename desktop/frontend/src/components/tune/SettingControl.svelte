<script lang="ts">
  import { appState } from '../../stores/appState.svelte';
  let { field, baseline }: { field: any; baseline?: string } = $props();
  let value = $derived(appState.settings[field.key] ?? field.initial ?? '');
  let resetValue = $derived(baseline ?? field.initial ?? '');
  function save(value: string) { appState.queueSaveSetting(field.key, value); }
</script>
<div class:wide={field.multiline} class="space-y-2">
  <div class="flex items-center justify-between gap-3">
    <label for={'setting-' + field.key} class="text-sm font-medium text-slate-200">{field.label}</label>
    {#if String(value) !== String(resetValue)}<button title={baseline !== undefined ? 'Restore automatic value' : 'Restore default'} onclick={() => save(String(resetValue))} class="text-xs text-emerald-400">Reset</button>{/if}
  </div>
  {#if field.toggle}
    <label class="flex items-center gap-3 text-sm text-slate-300"><input id={'setting-' + field.key} type="checkbox" checked={value === '1'} onchange={e => save(e.currentTarget.checked ? '1' : '0')} class="accent-emerald-400 h-4 w-4" />{value === '1' ? 'On' : 'Off'}</label>
  {:else if field.options}
    <select id={'setting-' + field.key} value={value} onchange={e => save(e.currentTarget.value)}>{#each field.options as option}<option value={typeof option === 'string' ? option : option.value}>{typeof option === 'string' ? option : option.label}</option>{/each}</select>
  {:else if field.multiline}
    <textarea id={'setting-' + field.key} rows="3" value={value} onchange={e => save(e.currentTarget.value)} placeholder="Optional"></textarea>
  {:else}
    <input id={'setting-' + field.key} type={field.text ? 'text' : 'number'} value={value} min={field.min} max={field.max} step={field.step ?? 'any'} onchange={e => save(e.currentTarget.value)} placeholder={field.initial === '' ? 'Auto / not set' : ''} />
  {/if}
  {#if field.help}<p class="text-xs leading-relaxed text-slate-500">{field.help}</p>{/if}
</div>
<style>
  .wide { grid-column:1 / -1; }
  input:not([type=checkbox]), select, textarea { width:100%; border:1px solid #ffffff18; border-radius:10px; background:#0b101a; color:#dce5ef; padding:9px 12px; font-size:13px; }
  input:focus, select:focus, textarea:focus { outline:1px solid #34d399; }
</style>
