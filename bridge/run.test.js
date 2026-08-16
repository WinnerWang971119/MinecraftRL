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

const {
  parseBridgeConfig,
  parsePadOrigin,
  parsePadIndex,
  parseOpponentMode,
  parseChallengerUsername,
  parseBoolean,
  usernamesForPad,
} = require('./run');

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

// ===========================================================================
// EXHIBITION MODE (T3). bot.js has carried `opponentMode` / `challengerUsername`
// since T1, but run.js wired NEITHER — so the whole human-opponent seam was
// unreachable from a command line and the exhibition path could not be launched
// at all. These are the two flags that turn it on.
// ===========================================================================

test('parseBridgeConfig omits both exhibition keys when neither is given (the training path)', () => {
  // The default must stay ABSENT, not 'bot': an omitted key falls through to
  // DEFAULT_BOT_CONFIG, which is what keeps the no-flags path byte-identical.
  assert.deepEqual(parseBridgeConfig([], {}), {});
  assert.deepEqual(parseBridgeConfig([], { OPPONENT_MODE: '', CHALLENGER_USERNAME: '' }), {});
});

test('parseBridgeConfig wires --opponent-mode and --challenger-username through to ArenaBots', () => {
  assert.deepEqual(
    parseBridgeConfig(['--opponent-mode', 'human', '--challenger-username', 'classmate_1'], {}),
    { opponentMode: 'human', challengerUsername: 'classmate_1' },
  );
  // The env form the launcher (T5) will use, and the CLI-wins precedence.
  assert.deepEqual(
    parseBridgeConfig([], { OPPONENT_MODE: 'human', CHALLENGER_USERNAME: 'classmate_2' }),
    { opponentMode: 'human', challengerUsername: 'classmate_2' },
  );
  assert.deepEqual(
    parseBridgeConfig(['--challenger-username', 'from_cli'], {
      OPPONENT_MODE: 'human',
      CHALLENGER_USERNAME: 'from_env',
    }),
    { opponentMode: 'human', challengerUsername: 'from_cli' },
  );
});

test('--challenger-username has NO reserved values: "auto" pins a player called auto', () => {
  // This flag used to read the literal `auto` as "no pin". `auto` is a legal
  // Minecraft username, so that made one classmate impossible to pin — while
  // `AUTO` pinned fine, the check being case-sensitive. T15 documents pinning
  // as THE demo-day mitigation for the bystander bug, so a pin that silently
  // degrades to "whoever walks in first" is the one failure this flag must not
  // have. Omitting the flag is the no-pin form.
  assert.deepEqual(
    parseBridgeConfig(['--opponent-mode', 'human', '--challenger-username', 'auto'], {}),
    { opponentMode: 'human', challengerUsername: 'auto' },
  );
  assert.deepEqual(parseBridgeConfig([], { OPPONENT_MODE: 'human', CHALLENGER_USERNAME: 'auto' }), {
    opponentMode: 'human',
    challengerUsername: 'auto',
  });
  // The no-pin form: the key stays ABSENT so it falls through to
  // DEFAULT_BOT_CONFIG's null, which bot.js reads as the first-claimant latch.
  // An empty env var is already read as unset, so a launcher that always
  // exports CHALLENGER_USERNAME can still say "no pin".
  assert.deepEqual(parseBridgeConfig(['--opponent-mode', 'human'], {}), {
    opponentMode: 'human',
  });
  assert.deepEqual(parseBridgeConfig([], { OPPONENT_MODE: 'human', CHALLENGER_USERNAME: '' }), {
    opponentMode: 'human',
  });
  // ...but an explicitly empty flag is a mistake, not a way to say "no pin".
  assert.throws(
    () => parseBridgeConfig(['--opponent-mode', 'human', '--challenger-username', ''], {}),
    /--challenger-username must be a Minecraft username/,
  );
});

test('parseBridgeConfig refuses a typo\'d opponent mode rather than falling back to training', () => {
  // The silent direction is what matters: `--opponent-mode humans` accepted as
  // a default would spend the whole exhibition fighting a dummy bot nobody
  // came to see, with no error anywhere.
  for (const bad of ['humans', 'HUMAN', 'Human', 'dummy', '', 'true']) {
    assert.throws(
      () => parseBridgeConfig(['--opponent-mode', bad], {}),
      /--opponent-mode must be "bot" or "human"/,
      `--opponent-mode ${bad} must be rejected`,
    );
  }
  assert.throws(
    () => parseBridgeConfig([], { OPPONENT_MODE: 'nope' }),
    /OPPONENT_MODE must be "bot" or "human"/,
  );
});

test('parseBridgeConfig refuses a challenger name that could never match a player', () => {
  // A pin is compared against `entity.username`, so anything outside the
  // username grammar can never match — producing an exhibition in which nobody
  // is ever the opponent, which looks exactly like nobody having joined.
  for (const bad of ['bad name', 'seventeen_chars_x', 'a"b']) {
    assert.throws(
      () => parseBridgeConfig(['--opponent-mode', 'human', '--challenger-username', bad], {}),
      /--challenger-username must be a Minecraft username/,
      `--challenger-username ${bad} must be rejected`,
    );
  }
});

test('a pinned challenger with no --opponent-mode human is refused, not silently ignored', () => {
  // In 'bot' mode the opponent is the dummy and the pinned name is read by
  // NOTHING — so this misconfiguration produces a demo in which the agent
  // fights an invisible dummy while the operator believes they pinned the
  // challenger. The likeliest demo-day mistake there is, and free to catch.
  assert.throws(
    () => parseBridgeConfig(['--challenger-username', 'classmate_1'], {}),
    /requires --opponent-mode human/,
  );
  assert.throws(
    () => parseBridgeConfig(['--opponent-mode', 'bot', '--challenger-username', 'classmate_1'], {}),
    /requires --opponent-mode human/,
  );
  // `auto` is a username like any other now, so it is caught by that rule too —
  // it used to slip through as the "no pin" sentinel.
  assert.throws(
    () => parseBridgeConfig(['--challenger-username', 'auto'], {}),
    /requires --opponent-mode human/,
  );
});

// ===========================================================================
// PER-OPPONENT MOBILITY TOGGLE (T11c). bot.js has carried
// `dummyKnockbackImmune` since T11c's first half, and — exactly like the
// exhibition keys above — NOTHING wired it: no flag, no env var, and
// distributed/launcher.py passed seven flags, none of them this. So AC18 ("the
// scripted opponent takes knockback") was unreachable outside unit tests, and
// the exhibition path cannot stand in for it: exhibition runs opponentMode
// 'human', where the bridge's override branch never fires. Training is the
// only path that can exercise it, and this is the flag that turns it on.
// ===========================================================================

test('parseBridgeConfig omits dummyKnockbackImmune when it is not given (the training default)', () => {
  // ABSENT, not `true`: an omitted key falls through to DEFAULT_BOT_CONFIG,
  // which is what keeps the no-flags path byte-identical to today.
  assert.deepEqual(parseBridgeConfig([], {}), {});
  assert.deepEqual(parseBridgeConfig([], { DUMMY_KNOCKBACK_IMMUNE: '' }), {});
});

test('parseBridgeConfig wires --dummy-knockback-immune through to ArenaBots', () => {
  assert.deepEqual(parseBridgeConfig(['--dummy-knockback-immune', 'false'], {}), {
    dummyKnockbackImmune: false,
  });
  assert.deepEqual(parseBridgeConfig(['--dummy-knockback-immune', 'true'], {}), {
    dummyKnockbackImmune: true,
  });
  // The env form, and the CLI-wins precedence, matching every other spec.
  assert.deepEqual(parseBridgeConfig([], { DUMMY_KNOCKBACK_IMMUNE: 'false' }), {
    dummyKnockbackImmune: false,
  });
  assert.deepEqual(
    parseBridgeConfig(['--dummy-knockback-immune', 'true'], { DUMMY_KNOCKBACK_IMMUNE: 'false' }),
    { dummyKnockbackImmune: true },
  );
  // The `=` form parseFlags also accepts.
  assert.deepEqual(parseBridgeConfig(['--dummy-knockback-immune=false'], {}), {
    dummyKnockbackImmune: false,
  });
});

test('parseBridgeConfig refuses the shell truthiness idioms rather than guessing', () => {
  // The dangerous direction is `0`/`no` read as truthy: the operator's argv
  // would say "this opponent takes knockback" while every episode of an
  // overnight run trained against an immune, immobile dummy.
  for (const bad of ['0', '1', 'yes', 'no', 'TRUE', 'False', '', 'on']) {
    assert.throws(
      () => parseBridgeConfig(['--dummy-knockback-immune', bad], {}),
      /--dummy-knockback-immune must be "true" or "false"/,
      `--dummy-knockback-immune ${bad} must be rejected`,
    );
  }
  assert.throws(
    () => parseBridgeConfig([], { DUMMY_KNOCKBACK_IMMUNE: 'nope' }),
    /DUMMY_KNOCKBACK_IMMUNE must be "true" or "false"/,
  );
});

test('--dummy-knockback-immune false is refused in human mode, where it is read by nothing', () => {
  // The mirror of the pinned-challenger rule above. handleReset's override
  // branch is keyed on `_opponentIsBot()`, so in an exhibition this flag does
  // NOTHING — and a human already takes knockback and already walks. An
  // operator who passes it there has misread the flag.
  assert.throws(
    () => parseBridgeConfig(['--opponent-mode', 'human', '--dummy-knockback-immune', 'false'], {}),
    /is read by nothing in "human" mode/,
  );
  // `true` is the default, so passing it explicitly is redundant rather than a
  // misunderstanding — and refusing it would break a launcher that always
  // exports the variable.
  assert.deepEqual(
    parseBridgeConfig(['--opponent-mode', 'human', '--dummy-knockback-immune', 'true'], {}),
    { opponentMode: 'human', dummyKnockbackImmune: true },
  );
  // The combination that matters is legal and needs no --opponent-mode at all:
  // 'bot' is the default.
  assert.deepEqual(parseBridgeConfig(['--dummy-knockback-immune', 'false'], {}), {
    dummyKnockbackImmune: false,
  });
  assert.deepEqual(
    parseBridgeConfig(['--opponent-mode', 'bot', '--dummy-knockback-immune', 'false'], {}),
    { opponentMode: 'bot', dummyKnockbackImmune: false },
  );
});

test('the toggle composes with a full launcher argv without disturbing it', () => {
  // The exact shape distributed/launcher.py emits for a non-immune pad.
  assert.deepEqual(
    parseBridgeConfig(
      [
        '--port',
        '25565',
        '--bridge-port',
        '5556',
        '--pad-index',
        '1',
        '--pad-origin',
        '512,0',
        '--learner-username',
        'learner_1',
        '--dummy-username',
        'dummy_1',
        '--dummy-knockback-immune',
        'false',
      ],
      {},
    ),
    {
      port: 25565,
      bridgePort: 5556,
      padIndex: 1,
      padOriginX: 512,
      padOriginZ: 0,
      learnerUsername: 'learner_1',
      dummyUsername: 'dummy_1',
      dummyKnockbackImmune: false,
    },
  );
});

test('parsePadOrigin/parsePadIndex/usernamesForPad are exported for the launcher and tests', () => {
  assert.deepEqual(parsePadOrigin('512,1024', '--pad-origin'), { x: 512, z: 1024 });
  assert.equal(parsePadIndex('7', '--pad-index'), 7);
  assert.deepEqual(usernamesForPad(0), { learnerUsername: 'learner_bot', dummyUsername: 'dummy_bot' });
  assert.deepEqual(usernamesForPad(7), { learnerUsername: 'learner_7', dummyUsername: 'dummy_7' });
  assert.equal(parseOpponentMode('human', '--opponent-mode'), 'human');
  assert.equal(parseChallengerUsername('classmate_1', '--challenger-username'), 'classmate_1');
  // No reserved values: this returns a username or throws, never null.
  assert.equal(parseChallengerUsername('auto', '--challenger-username'), 'auto');
  assert.equal(parseChallengerUsername('AUTO', '--challenger-username'), 'AUTO');
  assert.equal(parseBoolean('false', '--dummy-knockback-immune'), false);
  assert.equal(parseBoolean(' true ', '--dummy-knockback-immune'), true);
});
