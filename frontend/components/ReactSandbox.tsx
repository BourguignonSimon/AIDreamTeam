/**
 * ReactSandbox — browser-based React component playground.
 *
 * Architecture
 * ────────────
 *
 *  ┌──────────────────────────────────────────────────────────────┐
 *  │  ReactSandbox (parent)                                        │
 *  │                                                               │
 *  │  ┌─────────────────────┐   postMessage    ┌──────────────┐   │
 *  │  │  CodeEditor         │ ──UPDATE_CODE──► │  iframe      │   │
 *  │  │  (textarea + valid) │                  │  sandbox-    │   │
 *  │  └─────────────────────┘                  │  shell.html  │   │
 *  │         ▲                                 └──────────────┘   │
 *  │         │ tokens                                             │
 *  │  ┌─────────────────────┐                                     │
 *  │  │  useSandboxStream   │ ◄── WebSocket /ws/codegen           │
 *  │  └─────────────────────┘                                     │
 *  └──────────────────────────────────────────────────────────────┘
 *
 * Isolation guarantees
 * ────────────────────
 * - iframe: sandbox="allow-scripts"  (no allow-same-origin)
 *   → null origin, no access to cookies / localStorage / IndexedDB
 * - CSP on sandbox-shell.html: connect-src 'none'
 *   → user code cannot fetch / XHR / open WebSockets
 * - Validation: @babel/parser checks JSX syntax *before* injecting into iframe
 *
 * Hot-reload
 * ──────────
 * Parent calls iframe.contentWindow.postMessage({type:"UPDATE_CODE", code})
 * whenever validated code changes. The shell re-renders the component in-place
 * — no iframe src reload, no blank flash.
 */

'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import { useSandboxStream, type StreamStatus } from '@/hooks/useSandboxStream';
import { validateJSX, type ValidationResult }  from '@/lib/jsxValidator';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Debounce delay (ms) before validating + hot-reloading on manual edits. */
const VALIDATE_DEBOUNCE_MS = 350;

/** URL of the WebSocket code-generation endpoint. */
const DEFAULT_WS_URL =
  typeof window !== 'undefined'
    ? `ws://${window.location.hostname}:8000/ws/codegen`
    : 'ws://localhost:8000/ws/codegen';

const PLACEHOLDER_CODE = `\
function App() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <h2>Hello from the Sandbox!</h2>
      <p>Type your component code on the left, or use the generator above.</p>
    </div>
  );
}`;

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: StreamStatus }) {
  const map: Record<StreamStatus, { label: string; cls: string }> = {
    idle:       { label: 'Idle',       cls: 'bg-gray-100 text-gray-600' },
    connecting: { label: 'Connecting', cls: 'bg-yellow-100 text-yellow-700' },
    streaming:  { label: 'Streaming…', cls: 'bg-blue-100 text-blue-700 animate-pulse' },
    complete:   { label: 'Done',       cls: 'bg-green-100 text-green-700' },
    error:      { label: 'Error',      cls: 'bg-red-100 text-red-700' },
  };
  const { label, cls } = map[status];
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

function ValidationBadge({ result }: { result: ValidationResult | null }) {
  if (!result) return null;
  if (result.valid) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700">
        <svg className="h-3 w-3" viewBox="0 0 12 12" fill="currentColor">
          <path d="M10 3L5 8.5 2 5.5l-1 1L5 10.5l6-7-1-0.5z" />
        </svg>
        Valid JSX
      </span>
    );
  }
  const { line, column, message } = result.errors[0] ?? {};
  const position = line != null ? ` (line ${line}${column != null ? `:${column}` : ''})` : '';
  return (
    <span
      title={message}
      className="inline-flex max-w-xs items-center gap-1 truncate rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700"
    >
      <svg className="h-3 w-3 flex-shrink-0" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 1a5 5 0 100 10A5 5 0 006 1zm-.75 2.75h1.5v3.5h-1.5v-3.5zm0 4.5h1.5v1.5h-1.5v-1.5z" />
      </svg>
      {message}{position}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface ReactSandboxProps {
  /** Override the default WebSocket URL. */
  wsUrl?: string;
  /** Initial code to show in the editor. */
  initialCode?: string;
}

export default function ReactSandbox({
  wsUrl = DEFAULT_WS_URL,
  initialCode = PLACEHOLDER_CODE,
}: ReactSandboxProps) {
  // ---- streaming state ----
  const { code: streamedCode, status, error: streamError, generate, reset } =
    useSandboxStream(wsUrl);

  // ---- editor state ----
  const [editorCode,  setEditorCode]  = useState(initialCode);
  const [validation,  setValidation]  = useState<ValidationResult | null>(null);
  const [sandboxReady, setSandboxReady] = useState(false);
  const [prompt,      setPrompt]      = useState('');
  const [renderError, setRenderError] = useState<string | null>(null);

  // ---- refs ----
  const iframeRef       = useRef<HTMLIFrameElement>(null);
  const debounceTimer   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastInjectedRef = useRef<string>('');

  // -------------------------------------------------------------------------
  // When streaming completes, copy streamed code into the editor.
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (status === 'complete' && streamedCode) {
      setEditorCode(streamedCode);
    }
  }, [status, streamedCode]);

  // -------------------------------------------------------------------------
  // Listen for messages from the iframe (SANDBOX_READY, RENDER_OK, RENDER_ERROR).
  // -------------------------------------------------------------------------

  useEffect(() => {
    function handleMessage(ev: MessageEvent) {
      if (!ev.data || typeof ev.data !== 'object') return;
      const msg = ev.data as { type: string; message?: string };

      switch (msg.type) {
        case 'SANDBOX_READY':
          setSandboxReady(true);
          break;
        case 'RENDER_OK':
          setRenderError(null);
          break;
        case 'RENDER_ERROR':
          setRenderError(msg.message ?? 'Unknown render error');
          break;
        default:
          break;
      }
    }

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  // -------------------------------------------------------------------------
  // Inject code into the iframe (hot-reload via postMessage).
  // -------------------------------------------------------------------------

  const injectCode = useCallback((code: string) => {
    const iframe = iframeRef.current;
    if (!iframe?.contentWindow) return;
    if (code === lastInjectedRef.current) return;

    lastInjectedRef.current = code;
    iframe.contentWindow.postMessage({ type: 'UPDATE_CODE', code }, '*');
  }, []);

  // -------------------------------------------------------------------------
  // Validate and inject when editorCode changes (debounced).
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);

    debounceTimer.current = setTimeout(() => {
      const result = validateJSX(editorCode);
      setValidation(result);

      if (result.valid && sandboxReady) {
        injectCode(editorCode);
      }
    }, VALIDATE_DEBOUNCE_MS);

    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [editorCode, sandboxReady, injectCode]);

  // -------------------------------------------------------------------------
  // Re-inject when the iframe becomes ready (late load).
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (sandboxReady && validation?.valid) {
      injectCode(editorCode);
    }
    // Only react to sandboxReady becoming true; editorCode/validation have
    // their own effect above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sandboxReady]);

  // -------------------------------------------------------------------------
  // Handlers
  // -------------------------------------------------------------------------

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    reset();
    generate(prompt.trim());
  };

  const handleEditorChange = (val: string) => {
    setEditorCode(val);
    // Also clear the render error so old messages don't linger while typing.
    setRenderError(null);
  };

  const handleIframeLoad = () => {
    // The iframe (re)loaded — sandbox will post SANDBOX_READY, but reset the
    // flag here as a safety net so we don't inject before it's ready.
    setSandboxReady(false);
    lastInjectedRef.current = '';
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const isGenerating = status === 'streaming' || status === 'connecting';

  return (
    <div className="flex h-full flex-col gap-4">
      {/* ── Prompt bar ── */}
      <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-white p-3 shadow-sm">
        <input
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !isGenerating && handleGenerate()}
          placeholder="Describe a React component… (e.g. 'build a todo list')"
          className="min-w-0 flex-1 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm outline-none focus:border-indigo-400 focus:bg-white focus:ring-1 focus:ring-indigo-400"
          disabled={isGenerating}
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating || !prompt.trim()}
          className="flex-shrink-0 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isGenerating ? 'Generating…' : 'Generate'}
        </button>
        <StatusBadge status={status} />
      </div>

      {/* ── Error banner (WS or render) ── */}
      {(streamError || renderError) && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          {streamError ?? renderError}
        </div>
      )}

      {/* ── Split pane: editor | preview ── */}
      <div className="flex min-h-0 flex-1 gap-4">
        {/* Editor panel */}
        <div className="flex w-1/2 flex-col rounded-xl border border-gray-200 bg-white shadow-sm">
          {/* Editor header */}
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Editor
            </span>
            <div className="flex items-center gap-2">
              <ValidationBadge result={validation} />
              {isGenerating && (
                <span className="text-xs text-gray-400">
                  {streamedCode.length} chars
                </span>
              )}
            </div>
          </div>

          {/* Textarea editor */}
          <textarea
            value={editorCode}
            onChange={e => handleEditorChange(e.target.value)}
            spellCheck={false}
            className="flex-1 resize-none rounded-b-xl bg-gray-50 p-4 font-mono text-sm leading-relaxed text-gray-800 outline-none focus:bg-white"
            placeholder="// Write your React component here…"
          />
        </div>

        {/* Preview panel */}
        <div className="flex w-1/2 flex-col rounded-xl border border-gray-200 bg-white shadow-sm">
          {/* Preview header */}
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Preview
            </span>
            <div className="flex items-center gap-2">
              {sandboxReady ? (
                <span className="flex items-center gap-1 text-xs text-green-600">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500" />
                  Sandbox ready
                </span>
              ) : (
                <span className="flex items-center gap-1 text-xs text-gray-400">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-gray-300" />
                  Loading…
                </span>
              )}
              {renderError && (
                <span className="text-xs text-red-500">Render error</span>
              )}
            </div>
          </div>

          {/* Isolated iframe */}
          <iframe
            ref={iframeRef}
            src="/sandbox-shell.html"
            title="React Sandbox Preview"
            onLoad={handleIframeLoad}
            /**
             * Security:
             *  allow-scripts    → JS execution (React, Babel, user code)
             *  no allow-same-origin → null origin: no localStorage, no cookies,
             *                         no IndexedDB, no parent-DOM access
             *  no allow-forms   → no form submissions
             *  no allow-popups  → no new windows
             */
            sandbox="allow-scripts"
            className="flex-1 rounded-b-xl border-0 bg-white"
          />
        </div>
      </div>
    </div>
  );
}
