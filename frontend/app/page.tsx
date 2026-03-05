/**
 * Root page — hosts the React Sandbox playground.
 *
 * The page itself is a Server Component; ReactSandbox is a Client Component
 * ('use client') because it needs WebSocket, useState, useEffect, etc.
 */

import ReactSandbox from '@/components/ReactSandbox';

export default function Home() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-50">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600">
            <svg
              className="h-5 w-5 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-900">React Sandbox</h1>
            <p className="text-xs text-slate-500">
              AI Dream Team · Isolated iframe · Live preview
            </p>
          </div>
          <div className="ml-auto flex items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
              CSP strict
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-indigo-400" />
              WebSocket streaming
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-amber-400" />
              Hot-reload
            </span>
          </div>
        </div>
      </header>

      {/* Main content — fills remaining height */}
      <main className="min-h-0 flex-1 p-4">
        <ReactSandbox />
      </main>
    </div>
  );
}
