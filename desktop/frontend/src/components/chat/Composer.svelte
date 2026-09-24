<script lang="ts">
  import { Send, Square, Trash2, Zap } from 'lucide-svelte';
  import { appState } from '../../stores/appState.svelte';

  let inputVal = $state('');
  let textareaEl = $state<HTMLTextAreaElement | null>(null);

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleSubmit() {
    if (appState.engine.generating) {
      appState.cancelChat();
      return;
    }
    if (!inputVal.trim()) return;
    const prompt = inputVal;
    inputVal = '';
    if (textareaEl) textareaEl.style.height = 'auto';
    appState.sendMessage(prompt);
  }

  function autoResize() {
    if (!textareaEl) return;
    textareaEl.style.height = 'auto';
    textareaEl.style.height = Math.min(200, textareaEl.scrollHeight) + 'px';
  }
</script>

<div class="px-4 pb-4 pt-2 bg-gradient-to-t from-[#07090e] via-[#07090e]/95 to-transparent">
  <div class="max-w-4xl mx-auto">
    <div class="relative rounded-2xl bg-[#111724] border border-white/10 shadow-2xl focus-within:border-emerald-500/50 focus-within:ring-1 focus-within:ring-emerald-500/20 transition-all">
      <textarea
        bind:this={textareaEl}
        bind:value={inputVal}
        onkeydown={handleKeydown}
        oninput={autoResize}
        placeholder={appState.activeModel ? `Message ${appState.activeModel.name}...` : 'Select a model to start chatting...'}
        rows={1}
        class="w-full bg-transparent text-slate-100 placeholder-slate-500 px-4 py-3.5 text-sm resize-none focus:outline-none max-h-48 leading-relaxed font-sans"
      ></textarea>

      <div class="flex items-center justify-between px-3 py-2 border-t border-white/5 text-xs text-slate-400">
        <div class="flex items-center gap-2">
          {#if appState.telemetry.ctxTotal}
            <div class="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-white/5 text-[11px] font-mono">
              <Zap size={11} class="text-emerald-400" />
              <span>{appState.telemetry.ctxUsed} / {appState.telemetry.ctxTotal} ctx</span>
              <span class="text-slate-500">({appState.ctxPct}%)</span>
            </div>
          {/if}

          {#if appState.conversation.length > 0}
            <button onclick={() => appState.page = 'tune'} class="text-xs text-emerald-400" title="Change output token limit">Output: {appState.settings.max_tokens === '-1' ? 'No limit' : `${appState.settings.max_tokens || 512} tokens`}</button>
            <button
              onclick={() => appState.newChat()}
              class="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-rose-500/10 hover:text-rose-400 text-slate-500 transition-colors text-[11px]"
              title="Start a new chat; keep this conversation in history"
            >
              <Trash2 size={12} />
              <span>New chat</span>
            </button>
          {/if}
        </div>

        <div class="flex items-center gap-2">
          <span class="text-[11px] text-slate-500 hidden sm:inline">Enter to send, Shift+Enter for newline</span>
          <button
            onclick={handleSubmit}
            disabled={!appState.engine.generating && !inputVal.trim()}
            class="flex items-center justify-center w-8 h-8 rounded-xl transition-all shadow-md {appState.engine.generating ? 'bg-rose-500 hover:bg-rose-600 text-white animate-pulse' : 'bg-emerald-500 hover:bg-emerald-400 text-black disabled:opacity-40 disabled:hover:bg-emerald-500 disabled:cursor-not-allowed'}"
            title={appState.engine.generating ? 'Stop generation' : 'Send message'}
          >
            {#if appState.engine.generating}
              <Square size={13} fill="currentColor" />
            {:else}
              <Send size={14} class="translate-x-[-1px] translate-y-[1px]" />
            {/if}
          </button>
        </div>
      </div>
    </div>
  </div>
</div>
