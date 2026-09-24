import { marked } from 'marked';
import hljs from 'highlight.js';

marked.setOptions({
  breaks: true,
  gfm: true,
});

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning?: string;
  stats?: {
    tokSec?: number;
    ttftMs?: number;
    promptTokens?: number;
    genTokens?: number;
    stopReason?: string;
  };
}

export interface ModelItem {
  id: string;
  name: string;
  architecture: string;
  sizeBytes: number;
  layers: number;
  context: number;
  experts: number;
  projector: string;
  selected: boolean;
  valid: boolean;
  error?: string;
}

export interface ReviewBudget {
  label: string;
  kind: string;
  mib: number;
  note: string;
  device: 'vram' | 'ram';
}

export interface LogEntry {
  seq: number;
  text: string;
  level: 'info' | 'warn' | 'err';
}

export interface ToastItem {
  id: string;
  text: string;
  level: 'info' | 'success' | 'warn' | 'error';
}

export class AppState {
  // Svelte 5 runes mode: plain class fields are NOT reactive — every field the
  // UI reads must be $state, and every computed value must be $derived.
  page = $state<'chat' | 'models' | 'tune' | 'monitor'>('chat');
  settingsOpen = $state(false);

  app = $state({ name: 'LlamaCPP P100', version: '0.4.0', dataDir: '' });
  engine = $state({
    state: 'stopped' as 'stopped' | 'loading' | 'ready' | 'error',
    detail: '',
    stage: '',
    percent: -1,
    pendingReload: false,
    generating: false,
    uptimeS: 0,
  });

  hardware = $state({
    gpu: 'Detecting GPU...',
    threads: 1,
    ramTotalMib: 16384,
    vramTotalMib: 0,
    free_vram: 0,
  });

  telemetry = $state({
    ramTotalMib: 16384,
    ramFreeMib: 8192,
    vramTotalMib: 0,
    vramFreeMib: 0,
    gpuName: '',
    guiWorkingSetMib: 38,
    guiPrivateMib: 42,
    ctxUsed: 0,
    ctxTotal: 4096,
    processing: false,
  });

  stats = $state({
    requestsTotal: 0,
    tokensIn: 0,
    tokensOut: 0,
    lastTokSec: 0,
    lastTtftMs: 0,
    ctxUsed: 0,
    ctxTotal: 4096,
    guiWorkingSetMib: 38,
    guiPrivateMib: 42,
  });

  library = $state<{ folder: string; models: ModelItem[] }>({
    folder: '',
    models: [],
  });

  settings = $state<Record<string, any>>({});
  loadFields = $state<any[]>([]);
  outputTokens = $state(0);
  savingSettings = $state(false);
  private pendingSettings: Record<string, string> = {};
  private saveChain: Promise<boolean> = Promise.resolve(true);
  review = $state<{ budget: ReviewBudget[]; warnings: string[] }>({
    budget: [],
    warnings: [],
  });

  chats = $state<{id: string; title: string}[]>([]);
  activeChat = $state('');
  conversation = $state<Message[]>([]);
  stream = $state<{
    active: boolean;
    content: string;
    reasoning: string;
    ttftMs: number | null;
    tokSec: number;
    started: number;
  }>({
    active: false,
    content: '',
    reasoning: '',
    ttftMs: null,
    tokSec: 0,
    started: 0,
  });

  logs = $state<LogEntry[]>([]);
  toasts = $state<ToastItem[]>([]);
  scanning = $state(false);

  private evtSource: EventSource | null = null;
  private saveTimer: any = null;

  activeModel = $derived(
    this.library.models.find(m => m.id === this.settings.model) || null
  );

  vramUsedMib = $derived(
    this.telemetry.vramTotalMib
      ? Math.max(0, this.telemetry.vramTotalMib - (this.telemetry.vramFreeMib || 0))
      : 0
  );

  vramPct = $derived(
    this.telemetry.vramTotalMib
      ? Math.min(100, Math.round((this.vramUsedMib / this.telemetry.vramTotalMib) * 100))
      : 0
  );

  ramUsedMib = $derived(
    this.telemetry.ramTotalMib
      ? Math.max(0, this.telemetry.ramTotalMib - (this.telemetry.ramFreeMib || 0))
      : 0
  );

  ramPct = $derived(
    this.telemetry.ramTotalMib
      ? Math.min(100, Math.round((this.ramUsedMib / this.telemetry.ramTotalMib) * 100))
      : 0
  );

  ctxPct = $derived(
    this.telemetry.ctxTotal
      ? Math.min(100, Math.round((this.telemetry.ctxUsed / this.telemetry.ctxTotal) * 100))
      : 0
  );

  init() {
    fetch('/api/settings/schema').then(r => r.json()).then(fields => this.loadFields = fields).catch(() => this.toast('Could not load settings controls. Restart the updated app.', 'error'));
    this.fetchState();
    this.connectEvents();
    if (typeof window !== 'undefined') {
      window.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          if (this.evtSource) {
            this.evtSource.close();
            this.evtSource = null;
          }
        } else {
          this.connectEvents();
          this.fetchState();
        }
      });
    }
  }

  async fetchState() {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error(res.statusText);
      const st = await res.json();
      if (st.app) this.app = st.app;
      if (st.engine) Object.assign(this.engine, st.engine);
      if (st.hardware) Object.assign(this.hardware, st.hardware);
      if (st.library) this.library = st.library;
      if (st.settings) this.settings = { ...st.settings, ...this.pendingSettings };
      if (st.chats) this.chats = st.chats;
      if (st.activeChat) this.activeChat = st.activeChat;
      if (st.conversation) this.conversation = st.conversation;
      if (st.stats) {
        Object.assign(this.stats, st.stats);
        if (st.stats.guiWorkingSetMib) this.telemetry.guiWorkingSetMib = st.stats.guiWorkingSetMib;
      }
      if (st.review) this.review = st.review;
    } catch (e: any) {
      console.warn('fetchState error:', e);
    }
  }

  connectEvents() {
    if (this.evtSource || (typeof document !== 'undefined' && document.hidden)) return;
    this.evtSource = new EventSource('/api/events');
    this.evtSource.onopen = () => this.fetchState();
    this.evtSource.onerror = () => {};
    this.evtSource.onmessage = e => {
      try {
        const ev = JSON.parse(e.data);
        this.handleEvent(ev);
      } catch (_) {}
    };
  }

  handleEvent(ev: any) {
    switch (ev.type) {
      case 'hello':
        this.fetchState();
        break;
      case 'state':
        if (ev.engine) Object.assign(this.engine, ev.engine);
        break;
      case 'telemetry':
        Object.assign(this.telemetry, ev);
        break;
      case 'library':
      case 'settings':
      case 'conversation':
        this.fetchState();
        break;
      case 'scanning':
        this.scanning = !!ev.active;
        break;
      case 'toast':
        this.toast(ev.text, ev.level);
        break;
      case 'log':
        this.appendLog(ev);
        break;
      case 'chat.begin':
        this.outputTokens = 0;
        this.stream = {
          active: true,
          content: '',
          reasoning: '',
          ttftMs: null,
          tokSec: 0,
          started: Date.now(),
        };
        this.engine.generating = true;
        break;
      case 'chat.delta':
        if (ev.kind === 'reasoning') this.stream.reasoning += ev.text;
        else this.stream.content += ev.text;
        break;
      case 'chat.metrics':
        if (ev.ttftMs != null) this.stream.ttftMs = ev.ttftMs;
        this.outputTokens = ev.predN ?? this.outputTokens;
        this.stream.tokSec = ev.tokSec || 0;
        this.stats.lastTokSec = ev.tokSec || 0;
        break;
      case 'chat.done':
        this.finalizeChat(ev);
        break;
    }
  }

  appendLog(ev: any) {
    let level: 'info' | 'warn' | 'err' = 'info';
    if (/\bERR\b|error|failed|exception/i.test(ev.text)) level = 'err';
    else if (/\bWRN\b|warn/i.test(ev.text)) level = 'warn';

    this.logs.push({ seq: ev.seq, text: ev.text, level });
    if (this.logs.length > 3000) {
      this.logs.splice(0, this.logs.length - 3000);
    }
  }

  async sendMessage(prompt: string) {
    if (!prompt.trim() || this.engine.generating) return;
    if (!await this.flushSettings()) return;
    if (this.engine.state !== 'ready' || this.engine.pendingReload) {
      this.toast('Load or reload the model before sending a message.', 'info');
      return;
    }
    this.conversation.push({ role: 'user', content: prompt });
    this.stream = {
      active: true,
      content: '',
      reasoning: '',
      ttftMs: null,
      tokSec: 0,
      started: Date.now(),
    };
    this.engine.generating = true;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: prompt }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || res.statusText);
      }
    } catch (e: any) {
      this.toast(e.message, 'error');
      this.engine.generating = false;
      this.stream.active = false;
    }
  }

  finalizeChat(ev: any) {
    if (this.stream.content || this.stream.reasoning) {
      this.conversation.push({
        role: 'assistant',
        content: ev.content || this.stream.content,
        reasoning: ev.reasoning || this.stream.reasoning,
        stats: {
          tokSec: ev.tokSec ?? this.stream.tokSec,
          ttftMs: ev.ttftMs || this.stream.ttftMs,
          promptTokens: ev.usage?.prompt_tokens,
          genTokens: ev.usage?.completion_tokens,
          stopReason: ev.stopReason,
        },
      });
    }
    this.stream = {
      active: false,
      content: '',
      reasoning: '',
      ttftMs: null,
      tokSec: 0,
      started: 0,
    };
    this.engine.generating = false;
    this.fetchState();
  }

  async cancelChat() {
    try {
      await fetch('/api/chat/cancel', { method: 'POST' });
    } catch (e: any) {
      this.toast(e.message, 'error');
    }
  }

  async newChat(id?: string) {
    if (this.engine.generating) return;
    try {
      const res = await fetch(id ? '/api/conversation/select' : '/api/conversation/new', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Could not open chat');
      this.stream.active = false;
      this.page = 'chat';
      await this.fetchState();
    } catch (e: any) { this.toast(e.message, 'error'); }
  }

  async clearConversation() { await this.newChat(); }

  async selectModel(id: string) {
    if (!await this.flushSettings()) return;
    if (this.engine.generating || this.engine.state === 'loading') {
      this.toast('Wait for the current operation to finish before switching models.', 'info');
      return;
    }
    try {
      const res = await fetch('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to select model');
      this.settings.model = id;
      if (data.engine) Object.assign(this.engine, data.engine);
      await this.fetchState();
      if (this.engine.state !== 'ready' || this.engine.pendingReload) await this.loadEngine();
    } catch (e: any) {
      this.toast(e.message, 'error');
    }
  }

  async loadEngine() {
    if (!await this.flushSettings()) return;
    try {
      const res = await fetch('/api/engine/load', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load engine');
      if (data.engine) Object.assign(this.engine, data.engine);
      if (data.engine?.state === 'error') throw new Error(data.engine.detail || 'Failed to start engine');
      if (data.engine?.state === 'loading') this.toast('Engine starting...', 'info', 2500);
    } catch (e: any) {
      this.toast(e.message, 'error');
    }
  }

  async unloadEngine() {
    try {
      const res = await fetch('/api/engine/unload', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to unload engine');
      if (data.engine) Object.assign(this.engine, data.engine);
      this.toast('Engine stopped', 'info', 2500);
    } catch (e: any) {
      this.toast(e.message, 'error');
    }
  }

  async scanFolder(folder?: string) {
    if (this.scanning) return false;
    this.scanning = true;
    try {
      const res = await fetch('/api/models/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: folder || this.library.folder }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to scan folder');
      await this.fetchState();
      this.toast('Scanning models folder...', 'info', 2000);
      return true;
    } catch (e: any) {
      this.toast(e.message, 'error');
      this.scanning = false;
      return false;
    }
  }

  queueSaveSetting(key: string, value: string) {
    this.settings[key] = value;
    this.pendingSettings[key] = value;
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => this.flushSettings(), 400);
  }

  flushSettings(): Promise<boolean> {
    clearTimeout(this.saveTimer);
    const patch = { ...this.pendingSettings };
    this.pendingSettings = {};
    if (!Object.keys(patch).length) {
      const previous = this.saveChain;
      this.saveChain = previous.then(() => true);
      return previous;
    }
    this.saveChain = this.saveChain.then(async () => {
      this.savingSettings = true;
      try {
        const res = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(patch),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Could not save settings');
        if (data.engine) Object.assign(this.engine, data.engine);
        await this.fetchState();
        return true;
      } catch (e: any) {
        this.toast(e.message, 'error');
        await this.fetchState();
        return false;
      } finally {
        this.savingSettings = false;
      }
    });
    return this.saveChain;
  }

  async configureFreeToken(action: 'auto' | 'restore' | 'disable') {
    if (!await this.flushSettings()) return;
    try {
      const res = await fetch('/api/freetoken/configure', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Could not configure FreeToken');
      await this.fetchState();
      this.toast(action === 'auto' ? 'FreeToken configured for this machine' : action === 'restore' ? 'FreeToken baseline restored' : 'FreeToken disabled', 'success');
    } catch (e: any) { this.toast(e.message, 'error'); }
  }

  toast(text: string, level: 'info' | 'success' | 'warn' | 'error' = 'info', durationMs = 3500) {
    const id = Math.random().toString(36).substring(2, 9);
    this.toasts.push({ id, text, level });
    setTimeout(() => {
      this.toasts = this.toasts.filter(t => t.id !== id);
    }, durationMs);
  }
}

export const appState = new AppState();
