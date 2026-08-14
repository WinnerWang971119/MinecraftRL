// run.test.js — `node --test` suite for the per-pad config parsing in run.js.
// Runs WITHOUT a live Minecraft server: it drives the PURE
// parseBridgeConfig(argv, env) directly, so no Mineflayer, no socket, no bots.
//
// ============================================================================
// WHAT THIS FILE VERIFIES (testable now, no Paper server):
//   parseBridgeConfig (run.js):
//     - defaults: empty argv + empty env -> empty config (so ArenaBots uses the
//       DEFAULT_BOT_CONFIG 5555/25565/learner_bot/dummy_bot, byte-identical to
//       the single-arena path);
//     - CLI override: --port/--bridge-port/--learner-username/--dummy-username
//       land on the right keys, ports as Numbers;
//     - env override: MC_PORT/BRIDGE_PORT/LEARNER_USERNAME/DUMMY_USERNAME land
//       on the same keys;
//     - precedence: an explicit CLI flag wins over the matching env var;
//     - the optional --mc-host/--bridge-host extras are honored;
//     - invalid/missing values throw a clear Error (the script-level exit(1)
//       wraps this throw — tested as a throw so it is driveable offline).
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (NOT covered here):
//   The forwarded ports/usernames actually binding a second Paper server +
//   bridge on 25565+i / 5555+i — that is the human-verified launcher run (T11).
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { parseBridgeConfig, parsePadOrigin, parsePadIndex, usernamesForPad } = require('./run');

// ===========================================================================
// Defaults — nothing provided means an empty config (everything falls through
// to DEFAULT_BOT_CONFIG inside ArenaBots).
// ===========================================================================

test('parseBridgeConfig returns an empty config for empty argv + empty env', () => {
  const config = parseBridgeConfig([], {});
  assert.deepEqual(config, {}, 'no overrides -> all DEFAULT_BOT_CONFIG values');
});

test('parseBridgeConfig ignores an empty-string env var (treated as unset)', () => {
  // A launcher that exports BRIDGE_PORT="" must not coerce "" to a 0/NaN port.
  const config = parseBridgeConfig([], { MC_PORT: '', LEARNER_USERNAME: '' });
  assert.deepEqual(config, {}, 'blank env vars fall through to defaults');
});

// ===========================================================================
// CLI override — the four required flags, --flag value form.
// ===========================================================================

test('parseBridgeConfig reads the four required flags (--flag value form)', () => {
  const config = parseBridgeConfig(
    ['--port', '25566', '--bridge-port', '5556', '--learner-username', 'learner_1', '--dummy-username', 'dummy_1'],
    {},
  );
  assert.deepEqual(config, {
    port: 25566,
    bridgePort: 5556,
    learnerUsername: 'learner_1',
    dummyUsername: 'dummy_1',
  });
  assert.equal(typeof config.port, 'number', 'port is coerced to a Number');
  assert.equal(typeof config.bridgePort, 'number', 'bridgePort is coerced to a Number');
});

test('parseBridgeConfig accepts the --flag=value form too', () => {
  const config = parseBridgeConfig(
    ['--port=25570', '--bridge-port=5560', '--learner-username=learner_5', '--dummy-username=dummy_5'],
    {},
  );
  assert.deepEqual(config, {
    port: 25570,
    bridgePort: 5560,
    learnerUsername: 'learner_5',
    dummyUsername: 'dummy_5',
  });
});

test('parseBridgeConfig forwards only the keys provided (partial CLI override)', () => {
  const config = parseBridgeConfig(['--bridge-port', '5557'], {});
  assert.deepEqual(config, { bridgePort: 5557 }, 'only bridgePort set; the rest stay defaults');
});

test('parseBridgeConfig honors the optional --mc-host / --bridge-host extras', () => {
  const config = parseBridgeConfig(['--mc-host', '10.0.0.2', '--bridge-host', '0.0.0.0'], {});
  assert.deepEqual(config, { host: '10.0.0.2', bridgeHost: '0.0.0.0' });
});

// ===========================================================================
// Env override — the four required env vars (the form T11's launcher uses).
// ===========================================================================

test('parseBridgeConfig reads the four required env vars', () => {
  const config = parseBridgeConfig([], {
    MC_PORT: '25567',
    BRIDGE_PORT: '5557',
    LEARNER_USERNAME: 'learner_2',
    DUMMY_USERNAME: 'dummy_2',
  });
  assert.deepEqual(config, {
    port: 25567,
    bridgePort: 5557,
    learnerUsername: 'learner_2',
    dummyUsername: 'dummy_2',
  });
  assert.equal(typeof config.port, 'number');
  assert.equal(typeof config.bridgePort, 'number');
});

test('parseBridgeConfig trims surrounding whitespace on env values', () => {
  const config = parseBridgeConfig([], { MC_PORT: ' 25568 ', LEARNER_USERNAME: '  learner_3  ' });
  assert.deepEqual(config, { port: 25568, learnerUsername: 'learner_3' });
});

// ===========================================================================
// Precedence — an explicit CLI flag overrides the matching env var.
// ===========================================================================

test('parseBridgeConfig: a CLI flag overrides the matching env var', () => {
  const config = parseBridgeConfig(['--port', '25566', '--learner-username', 'cli_learner'], {
    MC_PORT: '25599',
    LEARNER_USERNAME: 'env_learner',
    // BRIDGE_PORT has no flag here -> the env value wins for that field.
    BRIDGE_PORT: '5558',
  });
  assert.deepEqual(config, {
    port: 25566, // CLI wins over MC_PORT=25599
    learnerUsername: 'cli_learner', // CLI wins over env
    bridgePort: 5558, // no flag -> env value used
  });
});

// ===========================================================================
// Invalid input — clear throws (the script wraps these as stderr + exit 1).
// ===========================================================================

test('parseBridgeConfig throws on a non-numeric port (--port foo)', () => {
  assert.throws(
    () => parseBridgeConfig(['--port', 'foo'], {}),
    /--port must be an integer port in 1\.\.65535/,
  );
});

test('parseBridgeConfig throws on a non-numeric env port', () => {
  assert.throws(
    () => parseBridgeConfig([], { BRIDGE_PORT: 'nope' }),
    /BRIDGE_PORT must be an integer port in 1\.\.65535/,
  );
});

test('parseBridgeConfig throws on a fractional, zero, negative, or out-of-range port', () => {
  assert.throws(() => parseBridgeConfig(['--port', '25566.5'], {}), /must be an integer port/);
  assert.throws(() => parseBridgeConfig(['--port', '0'], {}), /must be an integer port/);
  assert.throws(() => parseBridgeConfig(['--port', '-1'], {}), /must be an integer port/);
  assert.throws(() => parseBridgeConfig(['--bridge-port', '70000'], {}), /must be an integer port/);
});

test('parseBridgeConfig throws on an empty CLI username (--flag with a blank value)', () => {
  assert.throws(
    () => parseBridgeConfig(['--learner-username', '   '], {}),
    /--learner-username must be a non-empty value/,
  );
});

test('parseBridgeConfig throws on an unknown flag (a typo would silently use a default port)', () => {
  assert.throws(() => parseBridgeConfig(['--prot', '25566'], {}), /unknown flag "--prot"/);
});

test('parseBridgeConfig throws when a flag is missing its value', () => {
  assert.throws(() => parseBridgeConfig(['--port'], {}), /flag "--port" requires a value/);
  assert.throws(
    () => parseBridgeConfig(['--port', '--bridge-port', '5556'], {}),
    /flag "--port" requires a value/,
  );
});

test('parseBridgeConfig throws on a stray non-flag argument', () => {
  assert.throws(() => parseBridgeConfig(['25566'], {}), /unexpected argument "25566"/);
});

// ===========================================================================
// PAD TOPOLOGY (T9): --pad-origin / --pad-index.
//
// Both are PROCESS-LOCAL — neither reaches the wire. The anchor is only ever
// PARSED here; it is never derived from the index (padAnchor(i) is the
// launcher's sole implementation, deliberately not mirrored in this process).
//
// Validation is strict on purpose. The anchor is pasted TEXTUALLY into the
// datapack's macro arguments and the datapack builds `$(x).5` from it, so
// `512L` yields the non-coordinate `512L.5` and a bad value inside a $-macro
// aborts instantiation of the WHOLE function — no command runs and nothing
// reaches the server log. A NEGATIVE anchor is worse still: it parses fine and
// silently places the bots half a block off the anchor. Hence: two parts, each
// a plain non-negative integer, or a loud failure naming the offending string.
// ===========================================================================

test('parseBridgeConfig parses --pad-origin into the flat padOriginX/padOriginZ pair', () => {
  assert.deepEqual(parseBridgeConfig(['--pad-origin', '512,1024'], {}), {
    padOriginX: 512,
    padOriginZ: 1024,
  });
  assert.deepEqual(parseBridgeConfig(['--pad-origin=0,0'], {}), { padOriginX: 0, padOriginZ: 0 });
  // Surrounding and inner whitespace is tolerated; the VALUE grammar is not.
  assert.deepEqual(parseBridgeConfig([], { PAD_ORIGIN: ' 512 , 1024 ' }), {
    padOriginX: 512,
    padOriginZ: 1024,
  });
});

test('parseBridgeConfig fails loudly on a malformed --pad-origin, naming the offending value (TC19)', () => {
  const malformed = [
    '512', // one part
    '512,1024,0', // three parts
    '512,', // missing z
    ',1024', // missing x
    '512;1024', // wrong separator
    '-512,0', // NEGATIVE: the silent half-block misplacement
    '0,-1024',
    '512L,0', // NBT type suffix
    '0b,0',
    '512.0,0', // decimal
    '5e2,0', // exponent
    '0x10,0', // hex
    '+5,0', // signed
    '"512",0', // quoted
    'x,z',
    '',
    '   ',
  ];
  for (const value of malformed) {
    assert.throws(
      () => parseBridgeConfig(['--pad-origin', value], {}),
      (err) =>
        /^--pad-origin must be "<x>,<z>" with two non-negative plain integers/.test(err.message) &&
        err.message.includes(JSON.stringify(value)),
      `--pad-origin ${JSON.stringify(value)} must fail loudly with the offending value`,
    );
  }
  // The env form names its own source, and never defaults silently.
  assert.throws(
    () => parseBridgeConfig([], { PAD_ORIGIN: '512L,0' }),
    /PAD_ORIGIN must be "<x>,<z>" with two non-negative plain integers.*"512L,0"/,
  );
});

test('parseBridgeConfig derives the pad usernames from --pad-index, with i==0 keeping learner_bot/dummy_bot', () => {
  // i == 0 is DELIBERATELY learner_bot/dummy_bot (not learner_0), so the manual
  // single-arena path — ops.json, the arena:reset wrapper, the runbook — stays
  // byte-identical. This changes PR #21's launcher default.
  assert.deepEqual(parseBridgeConfig(['--pad-index', '0'], {}), {
    padIndex: 0,
    learnerUsername: 'learner_bot',
    dummyUsername: 'dummy_bot',
  });
  assert.deepEqual(parseBridgeConfig(['--pad-index', '3', '--pad-origin', '512,1024'], {}), {
    padIndex: 3,
    padOriginX: 512,
    padOriginZ: 1024,
    learnerUsername: 'learner_3',
    dummyUsername: 'dummy_3',
  });
  // Ports are NOT derived from the index: --pad-index is usernames and logging
  // only, and the launcher passes --bridge-port explicitly.
  assert.equal(parseBridgeConfig(['--pad-index', '3', '--pad-origin', '512,0'], {}).bridgePort, undefined);
});

test('parseBridgeConfig lets an explicit username override the pad-index default', () => {
  const config = parseBridgeConfig(
    ['--pad-index', '3', '--pad-origin', '512,0', '--learner-username', 'custom_learner'],
    { DUMMY_USERNAME: 'env_dummy' },
  );
  assert.equal(config.learnerUsername, 'custom_learner');
  assert.equal(config.dummyUsername, 'env_dummy', 'env also outranks the index-derived name');
});

test('parseBridgeConfig throws on a malformed --pad-index', () => {
  for (const value of ['-1', '1.5', '3L', 'three', '', ' ']) {
    assert.throws(
      () => parseBridgeConfig(['--pad-index', value], {}),
      (err) =>
        /^--pad-index must be a non-negative plain integer/.test(err.message) &&
        err.message.includes(JSON.stringify(value)),
      `--pad-index ${JSON.stringify(value)} must fail loudly`,
    );
  }
});

test('parseBridgeConfig refuses a nonzero --pad-index with no --pad-origin (silent pad overlap)', () => {
  // Defaulting the anchor here would stack pad 3 on top of pad 0: two bot pairs
  // in one arena, cross-crediting every hit with no attacker attribution. The
  // anchor must come from the launcher — this process never computes one.
  assert.throws(
    () => parseBridgeConfig(['--pad-index', '3'], {}),
    /--pad-index 3 requires an explicit --pad-origin/,
  );
  // Pad 0's anchor IS (0,0) by definition, so index 0 alone is fine.
  assert.doesNotThrow(() => parseBridgeConfig(['--pad-index', '0'], {}));
});

test('parseBridgeConfig rejects a username that could break the reset macro', () => {
  // Usernames are pasted between quotes inside the macro's NBT compound, so a
  // quote or comma would rewrite its argument list. Caught at STARTUP, not at
  // the first reset where the macro would abort silently server-side.
  assert.throws(
    () => parseBridgeConfig(['--learner-username', 'bad name'], {}),
    /learnerUsername must be a Minecraft username/,
  );
  assert.throws(
    () => parseBridgeConfig([], { DUMMY_USERNAME: 'a","b' }),
    /dummyUsername must be a Minecraft username/,
  );
});

test('parsePadOrigin/parsePadIndex/usernamesForPad are exported for the launcher and tests', () => {
  assert.deepEqual(parsePadOrigin('512,1024', '--pad-origin'), { x: 512, z: 1024 });
  assert.equal(parsePadIndex('7', '--pad-index'), 7);
  assert.deepEqual(usernamesForPad(0), { learnerUsername: 'learner_bot', dummyUsername: 'dummy_bot' });
  assert.deepEqual(usernamesForPad(7), { learnerUsername: 'learner_7', dummyUsername: 'dummy_7' });
});
