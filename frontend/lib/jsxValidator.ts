/**
 * JSX syntactic validator — client-side, zero server round-trips.
 *
 * Uses @babel/parser which runs entirely in the browser (it ships a UMD
 * build that Next.js can tree-shake and bundle).
 *
 * The validator wraps user code in the same IIFE structure that the sandbox
 * shell uses, so line numbers reported in errors align with what the user
 * sees in the editor.
 */

import * as BabelParser from '@babel/parser';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ValidationError {
  message: string;
  /** 1-based line in the *user* code (wrapper lines subtracted). */
  line: number | null;
  /** 1-based column. */
  column: number | null;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Number of lines prepended by the IIFE wrapper before user code. */
const WRAPPER_PREFIX_LINES = 11;

// ---------------------------------------------------------------------------
// Core validation
// ---------------------------------------------------------------------------

/**
 * Validate JSX source code without executing it.
 *
 * The function:
 * 1. Wraps the code in the same IIFE scaffold as the sandbox shell.
 * 2. Attempts to parse with Babel in JSX + ES2022 mode.
 * 3. Maps error positions back to user-visible line numbers.
 *
 * @param code  Raw JSX source as the user typed it.
 */
export function validateJSX(code: string): ValidationResult {
  if (!code.trim()) {
    return { valid: false, errors: [{ message: 'Empty code', line: null, column: null }] };
  }

  const wrapped = buildWrapped(code);

  try {
    BabelParser.parse(wrapped, parserOptions());
    return { valid: true, errors: [] };
  } catch (err: unknown) {
    return { valid: false, errors: [mapError(err)] };
  }
}

/**
 * Parse and return the AST (for advanced callers).
 * Throws a {@link ValidationError} on parse failure.
 */
export function parseJSX(code: string): ReturnType<typeof BabelParser.parse> {
  return BabelParser.parse(buildWrapped(code), parserOptions());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parserOptions(): BabelParser.ParserOptions {
  return {
    sourceType: 'script',
    plugins: [
      'jsx',
      'classProperties',
      'classStaticBlock',
      'objectRestSpread',
      'optionalChaining',
      'nullishCoalescingOperator',
      'logicalAssignment',
      'numericSeparator',
    ],
    errorRecovery: false,
    strictMode: false,
  };
}

/**
 * Build the same wrapper as the sandbox shell so error line numbers match.
 *
 * The wrapper accounts for WRAPPER_PREFIX_LINES lines above user code.
 */
function buildWrapped(code: string): string {
  return [
    '(function(__sandbox_render__, React) {',           // 1
    '  var useState      = React.useState;',            // 2
    '  var useEffect     = React.useEffect;',           // 3
    '  var useCallback   = React.useCallback;',         // 4
    '  var useMemo       = React.useMemo;',             // 5
    '  var useRef        = React.useRef;',              // 6
    '  var useContext    = React.useContext;',           // 7
    '  var useReducer    = React.useReducer;',          // 8
    '  var createContext = React.createContext;',        // 9
    '  var useId         = React.useId;',               // 10
    '',                                                  // 11  ← WRAPPER_PREFIX_LINES
    code,
    '',
    '  if (typeof App !== "undefined") {',
    '    __sandbox_render__(React.createElement(App, null));',
    '  }',
    '})',
  ].join('\n');
}

/** Map a raw Babel parse error back to user-space line numbers. */
function mapError(err: unknown): ValidationError {
  if (
    err &&
    typeof err === 'object' &&
    'message' in err
  ) {
    const raw = err as { message: string; loc?: { line?: number; column?: number } };
    const rawLine   = raw.loc?.line   ?? null;
    const rawColumn = raw.loc?.column ?? null;
    const userLine  = rawLine !== null ? Math.max(1, rawLine - WRAPPER_PREFIX_LINES) : null;

    // Strip the internal wrapper path Babel appends to the message
    const message = raw.message.replace(/\s*\([\d]+:[\d]+\)\s*$/, '').trim();

    return { message, line: userLine, column: rawColumn };
  }

  return { message: String(err), line: null, column: null };
}
