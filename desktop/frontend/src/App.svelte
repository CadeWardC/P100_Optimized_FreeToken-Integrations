<script lang="ts">
  import { onMount } from 'svelte';
  import { CheckCircle2, AlertTriangle, Info, XCircle } from 'lucide-svelte';
  import { appState } from './stores/appState.svelte';
  import Sidebar from './components/layout/Sidebar.svelte';
  import TopBar from './components/layout/TopBar.svelte';
  import ChatView from './components/chat/ChatView.svelte';
  import ModelsView from './components/models/ModelsView.svelte';
  import TuneView from './components/tune/TuneView.svelte';
  import MonitorView from './components/monitor/MonitorView.svelte';
  import SettingsModal from './components/settings/SettingsModal.svelte';

  onMount(() => {
    appState.init();
  });
</script>

<div class="flex h-screen w-screen overflow-hidden bg-[#07090e] text-slate-100 font-sans">
  <!-- Left Navigation Sidebar -->
  <Sidebar />

  <!-- Main Content Area -->
  <div class="flex-1 flex flex-col h-full overflow-hidden min-w-0">
    <!-- Top Bar -->
    <TopBar />

    <!-- Current Active Page -->
    <main class="flex-1 flex flex-col overflow-hidden min-w-0 relative">
      {#if appState.page === 'chat'}
        <ChatView />
      {:else if appState.page === 'models'}
        <ModelsView />
      {:else if appState.page === 'tune'}
        <TuneView />
      {:else if appState.page === 'monitor'}
        <MonitorView />
      {/if}
    </main>
  </div>

  <!-- Settings Modal Dialog -->
  {#if appState.settingsOpen}
    <SettingsModal />
  {/if}

  <!-- Floating Toast Notifications (bottom-right) -->
  <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
    {#each appState.toasts as toast (toast.id)}
      <div
        class="pointer-events-auto flex items-center gap-2.5 px-4 py-3 rounded-xl shadow-2xl border text-xs font-medium animate-in slide-in-from-bottom-2 fade-in {toast.level === 'error' ? 'bg-[#1a0f14] border-rose-500/30 text-rose-300' : toast.level === 'warn' ? 'bg-[#1c160f] border-amber-500/30 text-amber-300' : toast.level === 'success' ? 'bg-[#0e1c15] border-emerald-500/30 text-emerald-300' : 'bg-[#111724] border-white/10 text-slate-200'}"
      >
        {#if toast.level === 'error'}
          <XCircle size={15} class="text-rose-400 shrink-0" />
        {:else if toast.level === 'warn'}
          <AlertTriangle size={15} class="text-amber-400 shrink-0" />
        {:else if toast.level === 'success'}
          <CheckCircle2 size={15} class="text-emerald-400 shrink-0" />
        {:else}
          <Info size={15} class="text-blue-400 shrink-0" />
        {/if}
        <span class="leading-relaxed">{toast.text}</span>
      </div>
    {/each}
  </div>
</div>
