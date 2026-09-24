<script lang="ts">
  import { X, Check, Copy, Settings, FolderOpen } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';

  let serverVal = $state(appState.settings.server || '');
  let portVal = $state(appState.settings.port || '18080');
  let copied = $state(false);
  let saving = $state(false);
  let folderVal = $state(appState.library.folder);
  let browsing = $state(false);

  async function browseFolder() {
    browsing = true;
    try {
      const res = await fetch('/api/models/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: folderVal.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Could not open the folder picker');
      if (!data.cancelled) folderVal = data.folder;
    } catch (e: any) {
      appState.toast(e.message, 'error');
    } finally {
      browsing = false;
    }
  }

  function copyDir() {
    navigator.clipboard.writeText(appState.app.dataDir);
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }

  async function handleSave() {
    if (saving || browsing || appState.scanning) return;
    saving = true;
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server: serverVal.trim(), port: String(portVal).trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to save settings');
      appState.settings.server = serverVal.trim();
      appState.settings.port = String(portVal).trim();
      if (data.pendingReload) appState.engine.pendingReload = true;
      if (folderVal.trim() !== appState.library.folder) {
        if (!folderVal.trim()) throw new Error('Choose a model folder.');
        if (!await appState.scanFolder(folderVal.trim())) return;
      }
      appState.toast('Settings saved successfully', 'success');
      appState.settingsOpen = false;
    } catch (e: any) {
      appState.toast(e.message, 'error');
    } finally {
      saving = false;
    }
  }
</script>

<div class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in">
  <div class="w-full max-w-md max-h-[90vh] rounded-2xl bg-[#111724] border border-white/10 shadow-2xl overflow-hidden flex flex-col">
    <!-- Modal Header -->
    <div class="p-5 border-b border-white/5 flex items-center justify-between">
      <div class="flex items-center gap-2.5 font-bold text-slate-100 text-sm">
        <Settings size={18} class="text-emerald-400" />
        <span>Application Preferences</span>
      </div>
      <button
        onclick={() => appState.settingsOpen = false}
        class="p-1 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
      >
        <X size={16} />
      </button>
    </div>

    <!-- Modal Body -->
    <div class="p-6 space-y-4 text-xs font-sans overflow-y-auto">
      <div class="space-y-1.5">
        <label for="settings-model-folder" class="text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
          Model folder
        </label>
        <div class="flex gap-2">
          <input
            id="settings-model-folder"
            type="text"
            bind:value={folderVal}
            disabled={browsing || saving || appState.scanning}
            placeholder="Choose your models folder"
            spellcheck={false}
            aria-describedby="settings-model-folder-help"
            class="min-w-0 flex-1 px-3 py-2 bg-[#161d2e] border border-white/10 rounded-xl text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 font-mono disabled:opacity-50"
          />
          <button
            type="button"
            onclick={browseFolder}
            disabled={browsing || saving || appState.scanning}
            class="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 font-semibold disabled:opacity-50"
          >
            <FolderOpen size={14} />
            {browsing ? 'Choosing...' : 'Browse...'}
          </button>
        </div>
        <p id="settings-model-folder-help" class="text-slate-500 leading-relaxed">
          Browse or paste a path. Save Settings remembers the folder and scans it, including subfolders.
        </p>
      </div>
      <!-- Llama Server Executable -->
      <div class="space-y-1.5">
        <label for="settings-server-input" class="text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
          llama-server Executable Path
        </label>
        <input
          id="settings-server-input"
          type="text"
          bind:value={serverVal}
          placeholder="e.g. build/ft-windows-cuda-dev/bin/llama-server.exe"
          class="w-full px-3 py-2 bg-[#161d2e] border border-white/10 rounded-xl text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 font-mono"
        />
      </div>

      <!-- Local Port -->
      <div class="space-y-1.5">
        <div class="flex items-center justify-between">
          <label for="settings-port-input" class="text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
            Inference Server Port
          </label>
          <span class="text-[10px] text-amber-400 font-mono">restart required</span>
        </div>
        <input
          id="settings-port-input"
          type="number"
          bind:value={portVal}
          class="w-32 px-3 py-2 bg-[#161d2e] border border-white/10 rounded-xl text-xs text-slate-200 font-mono focus:outline-none focus:border-emerald-500/50"
        />
      </div>

      <!-- Data Folder -->
      <div class="space-y-1.5">
        <span class="text-slate-300 font-semibold uppercase tracking-wider text-[10px]">
          Storage Directory
        </span>
        <div class="flex items-center justify-between p-2.5 bg-[#161d2e] border border-white/10 rounded-xl text-xs font-mono text-slate-400">
          <span class="truncate mr-2">{appState.app.dataDir || 'LOCALAPPDATA/LlamaCppP100'}</span>
          <button
            onclick={copyDir}
            class="p-1 rounded hover:bg-white/10 text-slate-300 hover:text-white transition-colors"
            title="Copy path"
          >
            {#if copied}
              <Check size={13} class="text-emerald-400" />
            {:else}
              <Copy size={13} />
            {/if}
          </button>
        </div>
      </div>
    </div>

    <!-- Modal Footer -->
    <div class="p-4 bg-[#0d121c] border-t border-white/5 flex items-center justify-end gap-2">
      <button
        onclick={() => appState.settingsOpen = false}
        class="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
      >
        Cancel
      </button>
      <button
        onclick={handleSave}
        disabled={saving || browsing || appState.scanning}
        class="px-5 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black text-xs font-bold transition-all shadow-md shadow-emerald-500/10 disabled:opacity-50"
      >
        {saving ? 'Saving...' : 'Save Settings'}
      </button>
    </div>
  </div>
</div>
