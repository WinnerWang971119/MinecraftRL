// run.js — process entry for the bridge (RUNBOOK Step 0).
//
// Does the four-call wiring that takes ArenaBots live: construct, route
// inbound reset/step/close to the handlers, connect both bots, then open the
// TCP port for the Python env. Start Paper first, then `npm start`, then the
// Python driver.
//
// Two deliberate choices beyond the minimal wiring:
//   - Bots connect BEFORE listen(): the port only opens once the bridge can
//     actually serve a reset. If Paper is down, createBot fails and the
//     process exits 1 instead of opening a port that leads nowhere.
//   - 'error' listeners everywhere: BridgeServer and the Mineflayer bots are
//     EventEmitters, and an 'error' event with no listener crashes the
//     process. The M1 bar is zero crashes over >=100 episodes, so socket and
//     bot errors are logged and the Python side reconnects/retries.
//
// PER-PAD CONFIG (T9): one bridge process serves ONE pad, so to run N pads the
// launcher (T10) starts N of these, each with its own bridge port, usernames and
// pad ANCHOR. parseBridgeConfig reads those from argv/env and forwards them to
// ArenaBots; with no flags and no env the behavior is byte-identical to the
// single-arena default (MC port 25565, bridge port 5555, learner_bot/dummy_bot,
// anchor 0,0).
//
// Owner: RUNBOOK Step 0 (go-live wiring) / T9 (per-pad config)

'use strict';

// Only for isAbsolute() on --reset-request-path; nothing here builds a path.
const nodePath = require('node:path');

const {
  ArenaBots,
  assertMacroUsername,
  OPPONENT_MODE_BOT,
  OPPONENT_MODE_HUMAN,
} = require('./bot');

// Fire-and-forget Mineflayer calls (lookAt, attack) reject outside any await
// chain, and an unhandled rejection is process-fatal in Node — one killed the
// bridge mid-episode during the first live run. The M1 bar is zero crashes:
// log it, lose at worst one decision window, keep serving.
process.on('unhandledRejection', (reason) => {
  console.error('[bridge] unhandled rejection (continuing):', reason);
});

// ---------------------------------------------------------------------------
// Per-pad config parsing (T9). PURE: reads argv + env, returns a plain config
// object, no Mineflayer and no socket. Kept pure so `node --test` can drive it
// without a live server (see run.test.js).
//
// Only keys that were ACTUALLY provided are returned, so any key the caller
// omits falls through to bot.js DEFAULT_BOT_CONFIG (ArenaBots merges
// config over those defaults). That is what keeps the no-flags/no-env path
// byte-identical to today.
//
// Precedence per field: explicit CLI flag > env var > default (omitted here).
// Port values are coerced with Number(...) and must be finite positive
// integers; an invalid value throws a clear Error (main() turns that into a
// stderr message + exit 1) rather than silently falling back to a default.
// ---------------------------------------------------------------------------

/**
 * The argv flag <-> env var <-> config key mapping. Each entry declares how one
 * config field is sourced, via `kind`:
 *   'port'      finite integer in 1..65535 (the two ports);
 *   'text'      trimmed non-empty string (hosts, usernames);
 *   'padOrigin' "<x>,<z>" -> the padOriginX / padOriginZ pair;
 *   'padIndex'  a 0-based non-negative integer.
 *
 * Three more kinds were added below the original four, each with its own
 * parse* function and its own reason to be stricter than Number()/Boolean():
 *   'opponentMode'        exactly "bot" or "human";
 *   'challengerUsername'  a Minecraft username, no reserved values;
 *   'boolean'             exactly "true" or "false".
 *
 * Forwarded into ArenaBots(config) -> { ...DEFAULT_BOT_CONFIG, ...config }, so
 * the config keys here MUST match bot.js DEFAULT_BOT_CONFIG names exactly.
 */
const CONFIG_SPECS = Object.freeze([
  { key: 'port', flag: '--port', env: 'MC_PORT', kind: 'port' },
  { key: 'bridgePort', flag: '--bridge-port', env: 'BRIDGE_PORT', kind: 'port' },
  { key: 'learnerUsername', flag: '--learner-username', env: 'LEARNER_USERNAME', kind: 'text' },
  { key: 'dummyUsername', flag: '--dummy-username', env: 'DUMMY_USERNAME', kind: 'text' },
  // Optional extras for completeness; the four above are the required ones.
  { key: 'host', flag: '--mc-host', env: 'MC_HOST', kind: 'text' },
  { key: 'bridgeHost', flag: '--bridge-host', env: 'BRIDGE_HOST', kind: 'text' },
  // Pad topology (T9). PROCESS-LOCAL: neither value ever reaches the wire.
  { key: 'padOrigin', flag: '--pad-origin', env: 'PAD_ORIGIN', kind: 'padOrigin' },
  { key: 'padIndex', flag: '--pad-index', env: 'PAD_INDEX', kind: 'padIndex' },
  // EXHIBITION MODE (T3). Also process-local. Both keys existed in bot.js from
  // T1 but nothing wired them, so the whole human-opponent seam was dormant:
  // no flag, no env var, no way to reach it short of editing DEFAULT_BOT_CONFIG.
  // Omitting both reproduces the training path exactly ('bot' / null).
  { key: 'opponentMode', flag: '--opponent-mode', env: 'OPPONENT_MODE', kind: 'opponentMode' },
  {
    key: 'challengerUsername',
    flag: '--challenger-username',
    env: 'CHALLENGER_USERNAME',
    kind: 'challengerUsername',
  },
  // PER-OPPONENT MOBILITY TOGGLE (T11c). Same story as the two above: bot.js
  // has carried `dummyKnockbackImmune` since T11c's first half, but nothing a
  // real run starts could set it — no flag, no env var, and
  // distributed/launcher.py (the only thing that spawns training bridges)
  // passed seven flags, none of them this. AC18 is only exercisable on the
  // TRAINING path (exhibition runs opponentMode='human', where the override
  // branch is keyed off `_opponentIsBot()` and never fires), so without this
  // entry the scripted opponent stays knockback-immune and pinned to zero
  // movement speed in every run that matters. Omitting it reproduces the
  // training path exactly (true).
  {
    key: 'dummyKnockbackImmune',
    flag: '--dummy-knockback-immune',
    env: 'DUMMY_KNOCKBACK_IMMUNE',
    kind: 'boolean',
  },
  // IN-GAME CHAT RESET (demo day). Where deploy/exhibition.py's launcher polls
  // for a reset request, so a player typing the keyword in chat files exactly
  // the request `python -m deploy.exhibition --reset` files from a terminal.
  // Process-local like the rest; it never reaches the wire. Omitting it — which
  // every training launcher does — leaves bot.js at `resetRequestPath: null`
  // and the feature simply does not exist for that run.
  {
    key: 'resetRequestPath',
    flag: '--reset-request-path',
    env: 'RESET_REQUEST_PATH',
    kind: 'absolutePath',
  },
]);

/**
 * A NON-NEGATIVE PLAIN INTEGER, as a string, and nothing else.
 *
 * Deliberately stricter than Number(): the pad anchor is pasted TEXTUALLY into
 * the `arena:setup_pad` / `arena:reset_pad` macro arguments, and the datapack
 * builds `$(x).5` from it. `512L`, `0b`, `512.0`, `"512"` would each yield a
 * non-coordinate, and `-512` would silently land the bot half a block off the
 * anchor with no error at all — the dangerous case. A bad value inside a
 * `$`-macro aborts instantiation of the WHOLE function (no command in it runs,
 * nothing appears in the log), so it is validated HERE, at the boundary, rather
 * than trusted to complain downstream. `+5`, `5e2` and `0x10` are rejected for
 * the same reason: one canonical form. (Surrounding whitespace is trimmed off
 * before the test — a launcher's `"512, 1024"` is a formatting choice, not an
 * ambiguous value.)
 */
const PLAIN_NON_NEGATIVE_INT = /^\d+$/;

/** Render a raw value inside an error message without losing its type. */
function showRaw(raw) {
  return typeof raw === 'string' ? JSON.stringify(raw) : String(raw);
}

/**
 * Parse a `--pad-origin` value ("<x>,<z>") into the pad ANCHOR.
 *
 * The anchor is the learner SPAWN CELL, not the floor origin: learner feet land
 * at (x+0.5, 64, z+0.5) and the dummy at (x+3.5, 64, z+0.5). This function only
 * PARSES the value it is handed — it never derives an anchor from a pad index
 * (`padAnchor(i)` is T10's sole implementation and is deliberately not mirrored
 * here).
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error ('--pad-origin' / 'PAD_ORIGIN').
 * @returns {{x:number, z:number}} The parsed anchor.
 * @throws {Error} Naming the offending string, never defaulting silently.
 */
function parsePadOrigin(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  const requirement =
    `${source} must be "<x>,<z>" with two non-negative plain integers ` +
    `(e.g. "0,0" or "512,1024"; no sign, decimal point or NBT type suffix), got ${showRaw(raw)}`;
  if (typeof text !== 'string') {
    throw new Error(requirement);
  }
  const parts = text.split(',');
  if (parts.length !== 2) {
    throw new Error(requirement);
  }
  const coords = parts.map((part) => part.trim());
  for (const coord of coords) {
    if (!PLAIN_NON_NEGATIVE_INT.test(coord)) {
      throw new Error(requirement);
    }
  }
  const [x, z] = coords.map((coord) => Number(coord));
  // Beyond Number.MAX_SAFE_INTEGER the value stops round-tripping through
  // arithmetic (the +0.5 spawn offset in particular); a pad that far out is a
  // typo, not a topology.
  if (!Number.isSafeInteger(x) || !Number.isSafeInteger(z)) {
    throw new Error(requirement);
  }
  return { x, z };
}

/**
 * Parse a `--pad-index` value: a 0-based non-negative plain integer. Used for
 * usernames and logging only — never for coordinates.
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error.
 * @returns {number} The parsed index.
 */
function parsePadIndex(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  if (typeof text !== 'string' || !PLAIN_NON_NEGATIVE_INT.test(text)) {
    throw new Error(`${source} must be a non-negative plain integer, got ${showRaw(raw)}`);
  }
  const value = Number(text);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`${source} must be a non-negative plain integer, got ${showRaw(raw)}`);
  }
  return value;
}

/**
 * Parse an `--opponent-mode` value: exactly `bot` or `human` (T3).
 *
 * Validated HERE as well as in the ArenaBots constructor because the failure it
 * prevents is silent in the direction that matters: a typo'd `--opponent-mode
 * humans` on demo day must not fall back to the training path and spend the
 * exhibition swinging at a dummy bot nobody can see.
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error.
 * @returns {'bot'|'human'} The parsed mode.
 */
function parseOpponentMode(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  if (text !== OPPONENT_MODE_BOT && text !== OPPONENT_MODE_HUMAN) {
    throw new Error(
      `${source} must be "${OPPONENT_MODE_BOT}" or "${OPPONENT_MODE_HUMAN}", got ${showRaw(raw)}`,
    );
  }
  return text;
}

/**
 * Parse a `--challenger-username`: a Minecraft username, and nothing else.
 *
 * NO RESERVED VALUES — deliberately. This used to treat the literal `auto` as
 * "no pin", which meant a classmate actually called `auto` could never be
 * pinned, while `AUTO` could (the test was case-sensitive). Since T15 documents
 * pinning the challenger as THE demo-day mitigation for the bystander bug, a
 * pin that silently degrades to "whoever walks in first" is the one failure this
 * flag must not have. The no-pin form is OMITTING the flag: every unprovided key
 * falls through to DEFAULT_BOT_CONFIG's `challengerUsername: null`, which bot.js
 * reads as "the first non-own player in the pad claims the slot". An empty
 * `CHALLENGER_USERNAME` env var is already read as unset by parseBridgeConfig,
 * so a launcher that always exports the variable can say "no pin" with `''`;
 * an explicit `--challenger-username ""` is a mistake and still throws.
 *
 * Held to the same username grammar as the bot names even though this value
 * never reaches a datapack macro. It is compared against `entity.username`, so
 * anything that grammar rejects can never match a real player — and a pin that
 * can never match is an exhibition where nobody is ever the opponent, which
 * looks exactly like nobody having joined. Better to refuse it at startup.
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error.
 * @returns {string} The pinned username.
 */
function parseChallengerUsername(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  return assertMacroUsername(text, source);
}

/**
 * Parse a boolean flag/env value: exactly `true` or `false` (T11c).
 *
 * Held to parseOpponentMode's case-sensitive strictness, and for the same
 * reason — one canonical form, no silent fallback. The shell idioms a launcher
 * might reach for (`1`, `0`, `yes`, `TRUE`, an empty string) are all REJECTED
 * rather than guessed at: the dangerous direction is `--dummy-knockback-immune 0`
 * being read as truthy, which would leave the scripted opponent immune and
 * immobile for a whole overnight run while the argv says otherwise. (An empty
 * ENV var is read as unset by parseBridgeConfig before it reaches here, so a
 * launcher that always exports the variable can still say "use the default".)
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error.
 * @returns {boolean} The parsed value.
 */
function parseBoolean(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  if (text !== 'true' && text !== 'false') {
    throw new Error(`${source} must be "true" or "false", got ${showRaw(raw)}`);
  }
  return text === 'true';
}

/**
 * Parse a `--reset-request-path`: an ABSOLUTE filesystem path, and nothing else.
 *
 * The absoluteness is the entire point of validating this one. The launcher and
 * the bridge are two processes with DIFFERENT working directories — exhibition.py
 * spawns the bridge with `cwd=REPO_ROOT` while the launcher itself keeps
 * whatever directory the operator ran it from — so a relative path would have
 * them resolve two different files while agreeing character-for-character about
 * the string. The bridge would then file every chat reset into a file nobody
 * polls, with no error at either end and nothing in either log: a demo-day
 * keyword that answers "reset armed" in chat and does nothing at all. Refuse it
 * at the boundary instead, where the message can name the flag.
 *
 * @param {string} raw The raw flag/env value.
 * @param {string} source Human label for the error.
 * @returns {string} The path, trimmed.
 */
function parseAbsolutePath(raw, source) {
  const text = typeof raw === 'string' ? raw.trim() : raw;
  if (typeof text !== 'string' || text.length === 0 || !nodePath.isAbsolute(text)) {
    throw new Error(
      `${source} must be an ABSOLUTE path (the launcher and the bridge run in ` +
        `different working directories, so a relative one would name two ` +
        `different files), got ${showRaw(raw)}`,
    );
  }
  return text;
}

/**
 * The bot usernames implied by a pad index.
 *
 * `i == 0` is DELIBERATELY `learner_bot` / `dummy_bot` and not `learner_0`, so
 * the manual single-arena path (server/ops.json, the datapack's arena:reset
 * wrapper, every existing runbook step) stays byte-identical. This changes PR
 * #21's launcher default, which used `learner_0` at `i == 0`.
 *
 * @param {number} index 0-based pad index.
 * @returns {{learnerUsername:string, dummyUsername:string}}
 */
function usernamesForPad(index) {
  if (index === 0) {
    return { learnerUsername: 'learner_bot', dummyUsername: 'dummy_bot' };
  }
  return { learnerUsername: `learner_${index}`, dummyUsername: `dummy_${index}` };
}

/**
 * Parse the leading process arguments into a { flag: value } map. Accepts both
 * `--flag value` and `--flag=value`. Only the flags declared in CONFIG_SPECS
 * are recognized; an unknown `--flag` throws (a typo'd flag silently using a
 * default would launch an arena on the wrong port). A flag given with no value
 * (end of argv, or immediately followed by another flag) also throws.
 *
 * @param {string[]} argv Arguments AFTER the node + script entries (process.argv.slice(2)).
 * @returns {Map<string, string>} Flag-name -> raw string value.
 */
function parseFlags(argv) {
  const known = new Set(CONFIG_SPECS.map((spec) => spec.flag));
  const flags = new Map();
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (typeof token !== 'string' || !token.startsWith('--')) {
      throw new Error(`unexpected argument "${token}" (expected a --flag)`);
    }
    const eq = token.indexOf('=');
    if (eq !== -1) {
      // --flag=value form.
      const name = token.slice(0, eq);
      if (!known.has(name)) {
        throw new Error(`unknown flag "${name}"`);
      }
      flags.set(name, token.slice(eq + 1));
      continue;
    }
    // --flag value form: the value is the next token.
    if (!known.has(token)) {
      throw new Error(`unknown flag "${token}"`);
    }
    const value = argv[i + 1];
    if (value === undefined || (typeof value === 'string' && value.startsWith('--'))) {
      throw new Error(`flag "${token}" requires a value`);
    }
    flags.set(token, value);
    i += 1; // consume the value token
  }
  return flags;
}

/**
 * Coerce a raw port string to a finite positive integer, or throw a clear
 * error naming the source. TCP ports are 1..65535; reject anything outside that
 * (0 is not a usable bind target for a fixed per-pad port; the launcher assigns
 * bridge port 5555+i, while the Minecraft port stays 25565 — one JVM serves all
 * pads).
 *
 * @param {string} raw The raw value from a flag or env var.
 * @param {string} source A human label for the error (e.g. '--port' or 'MC_PORT').
 * @returns {number} The parsed port.
 */
function coercePort(raw, source) {
  const trimmed = typeof raw === 'string' ? raw.trim() : raw;
  const value = Number(trimmed);
  if (!Number.isInteger(value) || value < 1 || value > 65535) {
    throw new Error(`${source} must be an integer port in 1..65535, got "${raw}"`);
  }
  return value;
}

/**
 * Build the per-arena config from argv + env. Pure and side-effect free.
 *
 * @param {string[]} [argv] Arguments after node + script (default process.argv.slice(2)).
 * @param {object} [env] Environment map (default process.env).
 * @returns {object} A config object holding ONLY the keys that were provided,
 *   suitable for `new ArenaBots(config)`. Unspecified keys are absent so they
 *   fall through to DEFAULT_BOT_CONFIG.
 * @throws {Error} On an unknown/valueless flag or an invalid port value.
 */
function parseBridgeConfig(argv = process.argv.slice(2), env = process.env) {
  const flags = parseFlags(argv);
  const config = {};
  const provided = new Set();
  for (const spec of CONFIG_SPECS) {
    // Precedence: explicit CLI flag > env var > (omitted -> default).
    let raw;
    let source;
    if (flags.has(spec.flag)) {
      raw = flags.get(spec.flag);
      source = spec.flag;
    } else if (env && env[spec.env] !== undefined && env[spec.env] !== '') {
      raw = env[spec.env];
      source = spec.env;
    } else {
      continue; // not provided -> fall through to DEFAULT_BOT_CONFIG
    }
    provided.add(spec.key);

    if (spec.kind === 'port') {
      config[spec.key] = coercePort(raw, source);
    } else if (spec.kind === 'padOrigin') {
      // Flat keys, not a nested object: DEFAULT_BOT_CONFIG is only shallowly
      // frozen and ArenaBots merges with a spread, so a shared nested default
      // would be one mutation away from leaking across instances.
      const anchor = parsePadOrigin(raw, source);
      config.padOriginX = anchor.x;
      config.padOriginZ = anchor.z;
    } else if (spec.kind === 'padIndex') {
      config[spec.key] = parsePadIndex(raw, source);
    } else if (spec.kind === 'opponentMode') {
      config[spec.key] = parseOpponentMode(raw, source);
    } else if (spec.kind === 'challengerUsername') {
      config[spec.key] = parseChallengerUsername(raw, source);
    } else if (spec.kind === 'boolean') {
      config[spec.key] = parseBoolean(raw, source);
    } else if (spec.kind === 'absolutePath') {
      config[spec.key] = parseAbsolutePath(raw, source);
    } else {
      // Hosts / usernames pass through as trimmed strings. Reject an
      // all-whitespace value rather than connecting a blank username.
      const trimmed = typeof raw === 'string' ? raw.trim() : raw;
      if (typeof trimmed !== 'string' || trimmed.length === 0) {
        throw new Error(`${source} must be a non-empty value`);
      }
      config[spec.key] = trimmed;
    }
  }

  if (provided.has('padIndex')) {
    // A pad index with no anchor would silently stack pad i on top of pad 0 —
    // two bot pairs sharing one arena, cross-crediting every hit. The anchor is
    // NOT derived from the index here: padAnchor(i) is T10's sole
    // implementation and this process only ever parses what it is handed.
    if (!provided.has('padOrigin') && config.padIndex !== 0) {
      throw new Error(
        `--pad-index ${config.padIndex} requires an explicit --pad-origin "<x>,<z>" ` +
          '(the anchor comes from the launcher; defaulting it to 0,0 would stack this pad on pad 0)',
      );
    }
    // Usernames follow the index unless explicitly overridden.
    const implied = usernamesForPad(config.padIndex);
    if (!provided.has('learnerUsername')) {
      config.learnerUsername = implied.learnerUsername;
    }
    if (!provided.has('dummyUsername')) {
      config.dummyUsername = implied.dummyUsername;
    }
  }

  // Usernames are pasted into the reset macro's NBT arguments, so validate them
  // at STARTUP rather than at the first reset (where a bad name would abort the
  // whole macro function silently, server-side).
  for (const key of ['learnerUsername', 'dummyUsername']) {
    if (config[key] !== undefined) {
      assertMacroUsername(config[key], key);
    }
  }

  // A pinned challenger with no exhibition mode is INERT, not merely redundant:
  // in 'bot' mode the opponent is the dummy and the name is read by nothing at
  // all. Left to run, it produces a demo in which the agent fights a dummy the
  // audience cannot see while the operator believes they pinned the challenger.
  //
  // `!== undefined` is the whole test: providing the flag at all IS the pin now
  // that parseChallengerUsername has no reserved values to degrade into.
  if (
    config.challengerUsername !== undefined &&
    (config.opponentMode === undefined || config.opponentMode === OPPONENT_MODE_BOT)
  ) {
    throw new Error(
      `--challenger-username "${config.challengerUsername}" requires ` +
        `--opponent-mode ${OPPONENT_MODE_HUMAN} (a pinned challenger is read by nothing ` +
        `in "${OPPONENT_MODE_BOT}" mode, where the opponent is this pad's dummy bot)`,
    );
  }

  // The mirror image of the check above, and inert in the same silent way: the
  // override branch in handleReset is keyed on `_opponentIsBot()`, so asking for
  // a knockback-able opponent in 'human' mode does NOTHING. A human already
  // takes knockback and already walks — there is no dummy bot to un-pin — so an
  // operator who passes this expecting an effect has misread the flag, and the
  // exhibition would run with them believing they changed something.
  //
  // Only `false` is refused. `true` is the default: passing it explicitly in
  // 'human' mode is redundant rather than a misunderstanding, and refusing it
  // would break a launcher that always exports DUMMY_KNOCKBACK_IMMUNE=true.
  // Third of the same family, and inert in the same silent way. The chat-reset
  // handler is gated on 'human' mode (bot.js `_onChatMessage`), so a request
  // path handed to a TRAINING bridge is read by nothing: the operator would
  // believe the in-game keyword is armed while every `reset` typed into that
  // server did nothing. Refusing here is also the guard that keeps a future
  // fleet launcher from arming an in-game reset across 25 training arenas by
  // copying one flag too many out of the exhibition's argv.
  if (
    config.resetRequestPath !== undefined &&
    (config.opponentMode === undefined || config.opponentMode === OPPONENT_MODE_BOT)
  ) {
    throw new Error(
      `--reset-request-path "${config.resetRequestPath}" requires ` +
        `--opponent-mode ${OPPONENT_MODE_HUMAN} (the in-game chat reset is an ` +
        `exhibition feature and is read by nothing in "${OPPONENT_MODE_BOT}" mode)`,
    );
  }

  if (config.dummyKnockbackImmune === false && config.opponentMode === OPPONENT_MODE_HUMAN) {
    throw new Error(
      `--dummy-knockback-immune false is read by nothing in "${OPPONENT_MODE_HUMAN}" mode ` +
        '(there is no dummy bot to un-pin; a human challenger already takes knockback and ' +
        'moves at normal speed) — drop the flag, or drop --opponent-mode',
    );
  }

  return config;
}

async function main() {
  // Read per-arena overrides BEFORE any bot/socket work. An invalid value here
  // must fail fast and loud (exit 1), never silently bind a default port.
  let config;
  try {
    config = parseBridgeConfig();
  } catch (err) {
    console.error('[bridge] invalid config:', err.message);
    process.exit(1);
    return; // unreachable after exit, but keeps the type honest
  }

  // ArenaBots merges this over DEFAULT_BOT_CONFIG; an empty config reproduces
  // today's BridgeServer on 127.0.0.1:5555 with learner_bot/dummy_bot on 25565
  // at pad anchor (0,0).
  const bots = new ArenaBots(config);
  // In 'human' mode the second name is not a combatant: no dummy bot connects,
  // so print who the opponent actually is instead of a bot that will not exist.
  const opponent =
    bots.opponentMode === OPPONENT_MODE_BOT
      ? bots.config.dummyUsername
      : `human ${bots.challengerUsername === null ? '(first claimant in the pad)' : bots.challengerUsername}`;
  console.error(
    `[bridge] pad ${bots.padIndex} @ anchor ${bots.padOrigin.x},${bots.padOrigin.z} ` +
      `(${bots.config.learnerUsername} vs ${opponent})`,
  );
  // Said at startup, because the alternative is an operator discovering in
  // front of a room that the keyword was never armed. Silence means off.
  if (bots.resetRequestPath !== null) {
    console.error(
      `[bridge] in-game chat reset armed: a player typing "reset" files ` +
        `${bots.resetRequestPath}`,
    );
  }

  bots.transport.on('error', (err) => console.error('[bridge] transport error:', err));
  bots.transport.on('connection', () => console.error('[bridge] env connected'));
  bots.transport.on('disconnect', () => console.error('[bridge] env disconnected'));

  bots.wireTransport(); // route reset/step/close -> handlers

  await bots.connect(); // spawn learner_bot + dummy_bot (requires Paper up + opped)

  // `bots.dummy` is NULL for the whole run in 'human' mode — there is no second
  // connection to make — so this loop must skip it. Unguarded it threw on
  // `bot.on` and killed the bridge immediately after connect(), which is the
  // first thing an exhibition launch would have hit.
  for (const [name, bot] of [
    ['learner', bots.learner],
    ['dummy', bots.dummy],
  ]) {
    if (!bot) {
      continue;
    }
    bot.on('error', (err) => console.error(`[bridge] ${name} bot error:`, err));
    bot.on('kicked', (reason) => console.error(`[bridge] ${name} bot kicked:`, reason));
    bot.on('end', (reason) => console.error(`[bridge] ${name} bot disconnected:`, reason));
  }

  const { address, port } = await bots.transport.listen();
  console.error(`[bridge] listening on ${address}:${port}, both bots spawned`);

  process.once('SIGINT', () => {
    console.error('[bridge] SIGINT, shutting down');
    bots
      .close()
      .then(() => process.exit(0))
      .catch((err) => {
        // A teardown rejection (e.g. learner.quit() throwing) is otherwise
        // swallowed by the unhandledRejection handler above and process.exit
        // never runs, hanging the bridge until a second Ctrl-C. Exit anyway.
        console.error('[bridge] error during shutdown (exiting anyway):', err);
        process.exit(0);
      });
  });
}

// Only launch when run as a script (`node run.js`). When this module is
// require()d (run.test.js drives parseBridgeConfig with no live server),
// main() must NOT fire — it would try to spawn bots and open a socket.
if (require.main === module) {
  main().catch((err) => {
    console.error('[bridge] fatal:', err);
    process.exit(1);
  });
}

module.exports = {
  parseBridgeConfig,
  parsePadOrigin,
  parsePadIndex,
  parseOpponentMode,
  parseChallengerUsername,
  parseBoolean,
  parseAbsolutePath,
  usernamesForPad,
};
