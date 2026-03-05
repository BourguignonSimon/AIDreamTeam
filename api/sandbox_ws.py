"""WebSocket endpoint for React component code generation with token streaming.

Architecture
------------
- Client connects via WebSocket to ``/ws/codegen``
- Client sends a JSON message: ``{"prompt": "..."}``
- Server streams the generated JSX code token by token
- Each message has the form: ``{"type": "token", "content": "..."}``
- On completion: ``{"type": "done"}``
- On error: ``{"type": "error", "detail": "..."}``

In production, replace ``_stream_code_tokens`` with actual LLM API calls
(e.g. LiteLLM Proxy streaming or Anthropic Claude with stream=True).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Sample components library — indexed by intent keyword
# ---------------------------------------------------------------------------

_SAMPLE_COMPONENTS: dict[str, str] = {
    "counter": """\
function App() {
  const [count, setCount] = React.useState(0);

  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif', maxWidth: '400px' }}>
      <h1 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Counter</h1>
      <p style={{ fontSize: '3rem', fontWeight: 'bold', margin: '1rem 0', color: '#e94560' }}>
        {count}
      </p>
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          onClick={() => setCount(c => c - 1)}
          style={{ padding: '0.5rem 1.5rem', fontSize: '1.25rem', cursor: 'pointer',
                   borderRadius: '8px', border: '2px solid #e94560', background: 'white',
                   color: '#e94560', fontWeight: 'bold' }}
        >
          −
        </button>
        <button
          onClick={() => setCount(c => c + 1)}
          style={{ padding: '0.5rem 1.5rem', fontSize: '1.25rem', cursor: 'pointer',
                   borderRadius: '8px', border: 'none', background: '#e94560',
                   color: 'white', fontWeight: 'bold' }}
        >
          +
        </button>
        <button
          onClick={() => setCount(0)}
          style={{ padding: '0.5rem 1rem', fontSize: '0.9rem', cursor: 'pointer',
                   borderRadius: '8px', border: '1px solid #ccc', background: '#f5f5f5' }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
""",
    "todo": """\
function App() {
  const [items, setItems] = React.useState([
    { id: 1, text: 'Build something awesome', done: false },
    { id: 2, text: 'Ship to production', done: false },
  ]);
  const [input, setInput] = React.useState('');

  const addItem = () => {
    if (!input.trim()) return;
    setItems(prev => [...prev, { id: Date.now(), text: input.trim(), done: false }]);
    setInput('');
  };

  const toggle = (id) =>
    setItems(prev => prev.map(i => i.id === id ? { ...i, done: !i.done } : i));

  const remove = (id) =>
    setItems(prev => prev.filter(i => i.id !== id));

  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif', maxWidth: '480px' }}>
      <h1 style={{ color: '#1a1a2e', marginBottom: '1.5rem' }}>Todo List</h1>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addItem()}
          placeholder="Add a task..."
          style={{ flex: 1, padding: '0.5rem 0.75rem', borderRadius: '8px',
                   border: '2px solid #e2e8f0', fontSize: '0.95rem', outline: 'none' }}
        />
        <button
          onClick={addItem}
          style={{ padding: '0.5rem 1.25rem', background: '#6c63ff', color: 'white',
                   border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Add
        </button>
      </div>

      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {items.map(item => (
          <li key={item.id}
              style={{ display: 'flex', alignItems: 'center', gap: '0.75rem',
                       padding: '0.75rem', marginBottom: '0.5rem',
                       background: item.done ? '#f0fdf4' : '#fafafa',
                       borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <input
              type="checkbox"
              checked={item.done}
              onChange={() => toggle(item.id)}
              style={{ width: '1.1rem', height: '1.1rem', cursor: 'pointer' }}
            />
            <span style={{ flex: 1, textDecoration: item.done ? 'line-through' : 'none',
                           color: item.done ? '#6b7280' : '#1a1a2e' }}>
              {item.text}
            </span>
            <button
              onClick={() => remove(item.id)}
              style={{ background: 'none', border: 'none', cursor: 'pointer',
                       color: '#ef4444', fontSize: '1.1rem', padding: '0 0.25rem' }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      {items.length === 0 && (
        <p style={{ textAlign: 'center', color: '#9ca3af', fontStyle: 'italic' }}>
          No tasks yet. Add one above!
        </p>
      )}
    </div>
  );
}
""",
    "timer": """\
function App() {
  const [seconds, setSeconds] = React.useState(0);
  const [running, setRunning] = React.useState(false);
  const intervalRef = React.useRef(null);

  React.useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(() => setSeconds(s => s + 1), 1000);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [running]);

  const reset = () => {
    setRunning(false);
    setSeconds(0);
  };

  const pad = n => String(n).padStart(2, '0');
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;

  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif',
                  textAlign: 'center', maxWidth: '360px' }}>
      <h1 style={{ color: '#1a1a2e', marginBottom: '1.5rem' }}>Stopwatch</h1>
      <div style={{ fontSize: '4rem', fontWeight: 'bold', color: '#6c63ff',
                    letterSpacing: '0.05em', marginBottom: '2rem',
                    fontVariantNumeric: 'tabular-nums' }}>
        {pad(h)}:{pad(m)}:{pad(s)}
      </div>
      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
        <button
          onClick={() => setRunning(r => !r)}
          style={{ padding: '0.75rem 2rem', fontSize: '1rem', cursor: 'pointer',
                   borderRadius: '8px', border: 'none', fontWeight: 'bold',
                   background: running ? '#ef4444' : '#6c63ff', color: 'white' }}
        >
          {running ? 'Pause' : 'Start'}
        </button>
        <button
          onClick={reset}
          style={{ padding: '0.75rem 1.5rem', fontSize: '1rem', cursor: 'pointer',
                   borderRadius: '8px', border: '2px solid #e2e8f0',
                   background: 'white', color: '#374151' }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
""",
}

_DEFAULT_COMPONENT = _SAMPLE_COMPONENTS["counter"]


def _pick_component(prompt: str) -> str:
    """Select a sample component based on keywords in the prompt."""
    lower = prompt.lower()
    for keyword, code in _SAMPLE_COMPONENTS.items():
        if keyword in lower:
            return code
    return _DEFAULT_COMPONENT


# ---------------------------------------------------------------------------
# Token streaming
# ---------------------------------------------------------------------------

async def _stream_code_tokens(code: str, token_size: int = 4) -> AsyncIterator[str]:
    """Yield successive character chunks from *code* with a small async delay.

    In production replace this with streaming calls to LiteLLM / Anthropic.
    """
    for i in range(0, len(code), token_size):
        chunk = code[i : i + token_size]
        yield chunk
        await asyncio.sleep(0.015)  # ~67 tokens/s — adjust to taste


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

@router.websocket("/ws/codegen")
async def codegen_ws(websocket: WebSocket) -> None:
    """Stream React JSX code token by token to the browser.

    Protocol
    --------
    Client → Server::

        {"prompt": "build a counter"}

    Server → Client (repeated)::

        {"type": "token", "content": "function App"}

    Server → Client (once, on completion)::

        {"type": "done"}

    Server → Client (on error)::

        {"type": "error", "detail": "..."}
    """
    await websocket.accept()
    client = websocket.client
    logger.info('{"event":"ws_connect","client":"%s"}', client)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
                prompt = str(msg.get("prompt", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": "Invalid JSON — expected {\"prompt\": \"...\"}"}),
                )
                continue

            if not prompt:
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": "prompt must not be empty"}),
                )
                continue

            logger.info('{"event":"ws_codegen_start","prompt":%.80s}', prompt)
            component_code = _pick_component(prompt)

            try:
                async for token in _stream_code_tokens(component_code):
                    await websocket.send_text(
                        json.dumps({"type": "token", "content": token}),
                    )

                await websocket.send_text(json.dumps({"type": "done"}))
                logger.info('{"event":"ws_codegen_done","chars":%d}', len(component_code))

            except WebSocketDisconnect:
                logger.info('{"event":"ws_disconnect_during_stream","client":"%s"}', client)
                return

    except WebSocketDisconnect:
        logger.info('{"event":"ws_disconnect","client":"%s"}', client)
