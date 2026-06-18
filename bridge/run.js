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
// PER-ARENA CONFIG (T10): one bridge process serves ONE arena, so to run N
// arenas the launcher (T11) starts N of these, each with distinct ports and
// usernames. parseBridgeConfig reads those from argv/env and forwards them to
// ArenaBots; with no flags and no env the behavior is byte-identical to the
// single-arena default (MC port 25565, bridge port 5555, learner_bot/dummy_bot).
//
// Owner: RUNBOOK Step 0 (go-live wiring) / T10 (per-arena config)

'use strict';

const { ArenaBots } = require('./bot');

// Fire-and-forget Mineflayer calls (lookAt, attack) reject outside any await
// chain, and an unhandled rejection is process-fatal in Node — one killed the
// bridge mid-episode during the first live run. The M1 bar is zero crashes:
// log it, lose at worst one decision window, keep serving.
process.on('unhandledRejection', (reason) => {
  console.error('[bridge] unhandled rejection (continuing):', reason);
});

// ---------------------------------------------------------------------------
// Per-arena config parsing (T10). PURE: reads argv + env, returns a plain
// config object, no Mineflayer and no socket. Kept pure so `node --test` can
// drive it without a live server (see run.test.js).
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
 * config field is sourced. `port: true` marks a field that must parse as a
 * finite positive integer (the two ports); the rest are passed through as
 * trimmed strings (hosts, usernames).
 *
 * Forwarded into ArenaBots(config) -> { ...DEFAULT_BOT_CONFIG, ...config }, so
 * the config keys here MUST match bot.js DEFAULT_BOT_CONFIG names exactly.
 */
const CONFIG_SPECS = Object.freeze([
  { key: 'port', flag: '--port', env: 'MC_PORT', port: true },
  { key: 'bridgePort', flag: '--bridge-port', env: 'BRIDGE_PORT', port: true },
  { key: 'learnerUsername', flag: '--learner-username', env: 'LEARNER_USERNAME', port: false },
  { key: 'dummyUsername', flag: '--dummy-username', env: 'DUMMY_USERNAME', port: false },
  // Optional extras for completeness; the four above are the required ones.
  { key: 'host', flag: '--mc-host', env: 'MC_HOST', port: false },
  { key: 'bridgeHost', flag: '--bridge-host', env: 'BRIDGE_HOST', port: false },
]);

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
 * (0 is not a usable bind target for a fixed per-arena port, and the launcher
 * always assigns 5555+i / 25565+i).
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

    if (spec.port) {
      config[spec.key] = coercePort(raw, source);
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
  // today's BridgeServer on 127.0.0.1:5555 with learner_bot/dummy_bot on 25565.
  const bots = new ArenaBots(config);

  bots.transport.on('error', (err) => console.error('[bridge] transport error:', err));
  bots.transport.on('connection', () => console.error('[bridge] env connected'));
  bots.transport.on('disconnect', () => console.error('[bridge] env disconnected'));

  bots.wireTransport(); // route reset/step/close -> handlers

  await bots.connect(); // spawn learner_bot + dummy_bot (requires Paper up + opped)

  for (const [name, bot] of [['learner', bots.learner], ['dummy', bots.dummy]]) {
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

module.exports = { parseBridgeConfig };
