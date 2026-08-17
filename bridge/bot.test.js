// bot.test.js — `node --test` suite for the reset path. Runs WITHOUT a live
// Minecraft server.
//
// ============================================================================
// WHAT THIS FILE VERIFIES (testable now, no Paper server):
//   The datapack macro boundary (bot.js — T9):
//     - formatResetPadCommand / formatSetupPadCommand compose the macro calls
//       and REJECT anything that would abort the macro server-side (negative or
//       suffixed coordinates, non-plain usernames);
//     - at the default anchor the reset command is byte-identical to the
//       committed arena:reset pad-0 wrapper (AC11).
//   ArenaBots.handleReset (mock bots + mock transport):
//     - issues exactly ONE reset command — this pad's arena:reset_pad macro,
//       through the opped learner's chat — and replies reset_ack ok:true when
//       BOTH read-back gates match their templates INCLUDING the gear the
//       datapack gives (the exact configuration that failed live before
//       regearing was implemented);
//     - a STALE reset that loses the epoch race applies none of its four
//       post-gate effects, so a retry's live episode survives untouched;
//     - a dummy `death` fired during the reset window is discarded by the
//       winning handler's events.reset(), NOT by the suppression flag;
//     - a throwing bot.chat() rejects without acking and without stranding
//       the suppression flag, leaving the damage channel live;
//     - RESET CAUSALITY: both gates matching is NOT enough — the datapack's
//       per-bot beacon must also arrive, or a post-kill state that merely LOOKS
//       reset would be acked as one. Only this pad's beacon counts.
//   attack_cooldown across a reset (T18, issue #28):
//     - the first observation of an episode reports 0.0, not a phantom 1.0: the
//       reset's regear can re-zero the SERVER's attack-strength meter, so the
//       reported value ramps from the LATER of the last swing and the regear;
//     - the ramp reaches 1.0 at w4 and the anchor is re-armed every episode.
//   Static datapack contracts (read from the committed .mcfunction files):
//     - clear-before-give in both spawn functions, with no trailing clear (the
//       coverage the deleted _regear tests used to provide);
//     - the dummy is never given a weapon (the other half of the empty
//       dummyResetTemplate.inventory);
//     - server/world/datapacks/arena is in sync with server/arena.
//
// MOCK FIDELITY — READ BEFORE ADDING A FAKE HERE.
//   Mineflayer populates `health` ONLY on a bot's own connection; the entity
//   view of another player never carries it (prismarine-entity defines no such
//   field). The fakes below therefore put `health` on the BOT and never on
//   `entity`. A fake more capable than the real library is how the damage
//   channel shipped dead — see bridge/actions.test.js for the full note.
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (server/compat_check.md):
//   The datapack's /give landing in a real inventory within the gate's 3 s
//   timeout, and the dummy's real post-heal health landing within the dummy
//   gate.
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  ArenaBots,
  MAX_HEALTH,
  ACTION_REPEAT,
  OPPONENT_MODE_HUMAN,
  OPPONENT_HEALTH_UNAVAILABLE,
  PAD_INTERIOR_BOUNDS,
  isInsidePad,
  formatResetPadCommand,
  formatHumanResetCommands,
  formatSetupPadCommand,
  formatDeathObjectiveCommands,
  formatKnockbackResistanceCommand,
  formatMovementSpeedCommand,
  formatAttributeGetCommand,
  RL_DEATHS_OBJECTIVE,
  KNOCKBACK_RESISTANCE_ATTRIBUTE,
  KNOCKBACK_RESISTANCE_IMMUNE_VALUE,
  KNOCKBACK_RESISTANCE_NOT_IMMUNE_VALUE,
  MOVEMENT_SPEED_ATTRIBUTE,
  MOVEMENT_SPEED_STATIONARY_VALUE,
  MOVEMENT_SPEED_MOBILE_VALUE,
  ATTRIBUTE_GET_TRANSLATE_KEY,
  KNOCKBACK_RESISTANCE_NAME_KEY,
  MOVEMENT_SPEED_NAME_KEY,
  CHAT_RESET_CONFIRMATION,
  CHAT_RESET_DEBOUNCE_MS,
  matchesChatResetKeyword,
  formatChatResetRequest,
} = require('./bot');
// The REAL executor and the REAL weapon period drive the cooldown tests below:
// MacroExecutor owns lastSwingTick, so a hand-rolled stand-in would be testing
// the stand-in. IRON_SWORD_ATTACK_SPEED_TICKS is the one source of truth both
// modules share (bot.js imports it too), so the expected ramp cannot drift.
const { Macro, MacroExecutor, IRON_SWORD_ATTACK_SPEED_TICKS } = require('./actions');
// BridgeServer is the REAL receive path for the TC12 protocol-error test: bytes
// -> framer -> 'message' -> _handleMessage -> handleStep. A hand-rolled mock
// transport would prove nothing about what a live client can send.
const { validateOutbound, encodeMessage, BridgeServer } = require('./transport');
// Recorder spies live in the shared testkit so a change to the EventAggregator
// recorder surface cannot be applied here and silently missed in actions.test.js.
const { spyRecorders } = require('./testkit');

/** The learner spawn the reset template teleports to. */
const SPAWN = Object.freeze({ x: 0.5, y: 64.0, z: 0.5 });
/** The dummy spawn: the learner spawn offset +3 on x (see handleReset). */
const DUMMY_SPAWN = Object.freeze({ x: SPAWN.x + 3, y: SPAWN.y, z: SPAWN.z });

/** A minimal chat-capturing mock bot the regear/reset path can drive. */
function mockBot(username, { inventory = [], position = SPAWN } = {}) {
  const chatLog = [];
  return {
    username,
    chatLog,
    chat: (cmd) => chatLog.push(cmd),
    health: MAX_HEALTH,
    entity: {
      position: { x: position.x, y: position.y, z: position.z },
      effects: {},
    },
    inventory: { items: () => inventory.map((name) => ({ name })) },
  };
}

/**
 * A mock dummy that ALREADY satisfies its read-back gate: healed, at the +3 x
 * spawn, EMPTY-HANDED, no active effects. Without this the dummy gate
 * legitimately rejects the fake and burns its full 3 s timeout.
 *
 * The empty inventory is the template, not an omission: the datapack declares
 * the dummy "a passive target, no weapon" (spawn_dummy_pad.mcfunction only
 * /clear-s it) and the datapack is the sole reset authority, so nothing arms
 * the dummy any more. A fake holding a sword here would be a fake more capable
 * than the server — the mistake this suite exists to prevent.
 */
function mockDummy(overrides = {}) {
  return mockBot('dummy_bot', {
    inventory: [],
    position: DUMMY_SPAWN,
    ...overrides,
  });
}

/**
 * An EventEmitter-backed bot for the damage-channel reset tests — mineflayer
 * bots ARE EventEmitters, so `on`/`off`/`emit`/`listenerCount` behave exactly as
 * wireDamageEvents expects. `health` is on the bot only, never on `entity`.
 *
 * @param {string} username
 * @param {object} [opts]
 * @param {string[]} [opts.inventory] Items the gate reads back (mutable via setInventory).
 * @param {{x:number,y:number,z:number}} [opts.position]
 * @param {function} [opts.chat] Override to model a throwing/failing chat.
 */
function liveBot(username, { inventory = ['iron_sword'], position = SPAWN, chat } = {}) {
  const chatLog = [];
  const state = { inventory: [...inventory] };
  return Object.assign(new EventEmitter(), {
    username,
    chatLog,
    chat: chat || ((cmd) => chatLog.push(cmd)),
    health: MAX_HEALTH,
    entity: {
      position: { x: position.x, y: position.y, z: position.z },
      effects: {},
    },
    inventory: { items: () => state.inventory.map((name) => ({ name })) },
    /** Model the async /give finally landing in the real inventory. */
    setInventory(items) {
      state.inventory = [...items];
    },
  });
}

/**
 * Make an EventEmitter-backed bot pair answer a reset command the way the
 * server does: the datapack's spawn_learner_pad / spawn_dummy_pad each end with
 * a beacon addressed to their own bot, so both arrive on their own connection
 * when arena:reset_pad runs to completion.
 *
 * Tests that DON'T call this model a datapack that never spoke — which is the
 * silent-failure case the beacon exists to catch, so it must be opt-in here
 * rather than baked into the fixture.
 *
 * @param {object} arena An ArenaBots with EventEmitter learner + dummy.
 */
function answerResetLikeTheServer(arena, { delayMs = 5 } = {}) {
  const inner = arena.learner.chat;
  arena.learner.chat = (cmd) => {
    inner(cmd);
    const parsed = RESET_PAD_CALL.exec(typeof cmd === 'string' ? cmd : '');
    if (parsed === null) {
      return;
    }
    const [, x, z, learner, dummy, nonce] = parsed;
    // TIMING FIDELITY — this is the whole point of the fixture. A real beacon
    // is a ROUND TRIP: the chat command must reach the server, the macro must
    // instantiate and run, and the system_chat packet must come back and be
    // decoded. None of that can happen inside the synchronous span of
    // handleReset. An earlier version of this helper emitted both beacons
    // synchronously from chat(), which made a fake strictly more capable than
    // the server and hid a bug that double-failed every healthy reset. Deliver
    // it on a timer, and build the text from the ARGUMENTS THE MACRO RECEIVED
    // (exactly as $(x)/$(nonce) substitution does) rather than from the arena's
    // own expectation, so the test cannot agree with the code by construction.
    setTimeout(() => {
      arena.learner.emit('message', `[arena] reset_ok learner ${x} ${z} ${learner} ${nonce}`);
      arena.dummy.emit('message', `[arena] reset_ok dummy ${x} ${z} ${dummy} ${nonce}`);
    }, delayMs);
  };
}

/** The exact shape of the reset macro call the fixture answers. */
const RESET_PAD_CALL =
  /^\/function arena:reset_pad \{x:(\d+),z:(\d+),learner:"([A-Za-z0-9_]+)",dummy:"([A-Za-z0-9_]+)",nonce:(\d+)\}$/;

/** The shape of the read-back the override issues (T11c). */
const ATTRIBUTE_GET_CALL = /^\/attribute ([A-Za-z0-9_]+) (\S+) base get$/;

/**
 * The REAL prismarine-chat renderer, at the pinned protocol version, so the
 * `/attribute ... base get` fixture below builds its reply exactly the way
 * mineflayer will hand one to `_onBotMessage` — a ChatMessage, not a
 * hand-rolled object with a convenient toString(). A stand-in here would be a
 * fake more capable than the client (nothing would then prove the bridge can
 * read `with[2]` out of a JSON *number*), which is the failure mode this
 * suite's other fixtures are written to avoid. It costs ~90 ms to load;
 * mineflayer itself is never required by these tests.
 */
const ChatMessage = require('prismarine-chat')('1.21.1');

/**
 * The attribute-id -> description-id mapping the SERVER applies. Verified
 * against server/versions/1.21.1/paper-1.21.1.jar: AttributeCommand's
 * `getAttributeDescription` is `Component.translatable(attr.getDescriptionId())`
 * and 1.21.1's Attributes registers those ids as `attribute.name.generic.*`.
 */
const ATTRIBUTE_NAME_KEYS = Object.freeze({
  [KNOCKBACK_RESISTANCE_ATTRIBUTE]: KNOCKBACK_RESISTANCE_NAME_KEY,
  [MOVEMENT_SPEED_ATTRIBUTE]: MOVEMENT_SPEED_NAME_KEY,
});

/**
 * Answer the override's `/attribute ... base get` on the DUMMY's connection the
 * way Paper does (T11c).
 *
 * The component is the one AttributeCommand.getAttributeBase builds:
 * `Component.translatable("commands.attribute.base_value.get.success",
 *   getAttributeDescription(attr), entity.getName(), Double.valueOf(v))` — so
 * the value rides as a JSON NUMBER, not a string, which is exactly why the
 * bridge compares numerically instead of textually (`base set 0.0` comes back
 * through JSON.parse as `0`).
 *
 * Delivered on a timer for the same TIMING FIDELITY reason
 * `answerResetLikeTheServer` documents: a reply is a round trip and cannot land
 * inside the synchronous span of the command that asked for it.
 *
 * @param {object} arena An ArenaBots with an EventEmitter dummy.
 * @param {object} [opts]
 * @param {Record<string, number>} [opts.values] Override what the server reports
 *   per attribute id; defaults to the values the override actually sets.
 * @param {string[]} [opts.silentFor] Attribute ids to answer with NOTHING.
 * @param {number} [opts.delayMs]
 */
function answerAttributeGetsLikeTheServer(arena, { values = {}, silentFor = [], delayMs = 2 } = {}) {
  const applied = {
    [KNOCKBACK_RESISTANCE_ATTRIBUTE]: Number(KNOCKBACK_RESISTANCE_NOT_IMMUNE_VALUE),
    [MOVEMENT_SPEED_ATTRIBUTE]: Number(MOVEMENT_SPEED_MOBILE_VALUE),
    ...values,
  };
  const inner = arena.dummy.chat;
  arena.dummy.chat = (cmd) => {
    inner(cmd);
    const parsed = ATTRIBUTE_GET_CALL.exec(typeof cmd === 'string' ? cmd : '');
    if (parsed === null) {
      return;
    }
    const [, username, attribute] = parsed;
    if (silentFor.includes(attribute)) {
      return;
    }
    const nameKey = ATTRIBUTE_NAME_KEYS[attribute];
    setTimeout(() => {
      arena.dummy.emit(
        'message',
        new ChatMessage({
          translate: ATTRIBUTE_GET_TRANSLATE_KEY,
          with: [{ translate: nameKey }, { text: username }, applied[attribute]],
        }),
      );
    }, delayMs);
  };
}

/**
 * Read-back gate options that PARK the gate: `now` never advances (so the gate
 * cannot time out) and `sleep` hands back a promise the test releases by hand.
 * `parked` resolves the first time the gate actually sleeps, so tests never have
 * to guess at microtask timing.
 */
function parkedGate() {
  let release;
  const released = new Promise((resolve) => {
    release = resolve;
  });
  let signalParked;
  const parked = new Promise((resolve) => {
    signalParked = resolve;
  });
  return {
    parked,
    release: () => release(),
    options: {
      timeoutMs: 3000,
      pollIntervalMs: 1,
      now: () => 0,
      sleep: () => {
        signalParked();
        return released;
      },
    },
  };
}

// ===========================================================================
// THE DATAPACK MACRO BOUNDARY (T9).
//
// These replace the old `_regear` unit tests. `_regear` is gone: /clear +
// /give now live in the datapack's spawn_learner_pad / spawn_dummy_pad macros
// alongside the rest of the reset template, so there is exactly ONE place that
// decides what a bot holds at episode start (its clear-before-give ordering is
// preserved there and is load-bearing for the same-tick instant effects). What
// remains bridge-side, and what is pinned here, is the composition of the macro
// call itself — the point where an unvalidated value would abort the whole
// datapack function server-side with nothing in the log.
// ===========================================================================

test('formatResetPadCommand composes the macro call with quoted usernames', () => {
  assert.equal(
    formatResetPadCommand({ x: 512, z: 1024, learner: 'learner_3', dummy: 'dummy_3', nonce: 7 }),
    '/function arena:reset_pad {x:512,z:1024,learner:"learner_3",dummy:"dummy_3",nonce:7}',
  );
  assert.equal(formatSetupPadCommand({ x: 512, z: 1024 }), '/function arena:setup_pad {x:512,z:1024}');
});

test('the default-anchor reset command is byte-identical to the committed arena:reset wrapper (AC11)', () => {
  // The strongest N=1 identity proof available offline: read what the datapack
  // itself invokes for pad 0 and require the bridge to emit the same text. If
  // T6's wrapper or this formatter ever drifts, this fails.
  const wrapper = fs.readFileSync(
    path.join(__dirname, '..', 'server', 'arena', 'data', 'arena', 'function', 'reset.mcfunction'),
    'utf8',
  );
  const line = wrapper
    .split('\n')
    .map((raw) => raw.trim())
    .find((raw) => raw.startsWith('function arena:reset_pad'));
  assert.ok(line, 'arena:reset invokes arena:reset_pad');

  const bots = new ArenaBots({}, { transport: { send: () => {} } });
  // The only differences are the leading `/` that makes it a chat command and
  // the per-reset nonce, which the pad-0 wrapper pins to 0 (a datapack file
  // cannot carry a value that changes every episode).
  assert.equal(bots._resetPadCommand(0), `/${line}`);
  assert.equal(
    bots._resetPadCommand(0),
    '/function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot",nonce:0}',
  );
});

test('macro coordinates reject anything that would abort or silently mis-place the pad', () => {
  // A negative anchor is the DANGEROUS case: `$(x).5` would expand to `-512.5`,
  // i.e. anchor MINUS half a block, with no server-side error at all.
  for (const bad of [-1, -512, 1.5, NaN, Infinity, '0', '512L', null, undefined]) {
    assert.throws(
      () => formatResetPadCommand({ x: bad, z: 0, learner: 'learner_bot', dummy: 'dummy_bot', nonce: 0 }),
      /pad anchor x must be a non-negative plain integer/,
      `x=${String(bad)} must be rejected`,
    );
    assert.throws(
      () => formatSetupPadCommand({ x: 0, z: bad }),
      /pad anchor z must be a non-negative plain integer/,
      `z=${String(bad)} must be rejected`,
    );
  }
});

test('macro usernames reject anything that could rewrite the NBT argument list', () => {
  for (const bad of ['', 'bad name', 'quote"name', 'a,b', 'x}', 'seventeen_chars_x', 42, null]) {
    assert.throws(
      () => formatResetPadCommand({ x: 0, z: 0, learner: bad, dummy: 'dummy_bot', nonce: 0 }),
      /learner username must be a Minecraft username/,
      `learner=${String(bad)} must be rejected`,
    );
    assert.throws(
      () => formatResetPadCommand({ x: 0, z: 0, learner: 'learner_bot', dummy: bad, nonce: 0 }),
      /dummy username must be a Minecraft username/,
      `dummy=${String(bad)} must be rejected`,
    );
  }
});

test('an ArenaBots built with a malformed anchor or username fails at construction', () => {
  assert.throws(
    () => new ArenaBots({ padOriginX: -512 }, { transport: { send: () => {} } }),
    /pad anchor x \(--pad-origin\) must be a non-negative plain integer, got -512/,
  );
  assert.throws(
    () => new ArenaBots({ padIndex: 1.5 }, { transport: { send: () => {} } }),
    /padIndex \(--pad-index\) must be a non-negative plain integer, got 1.5/,
  );
  assert.throws(
    () => new ArenaBots({ learnerUsername: 'bad name' }, { transport: { send: () => {} } }),
    /learnerUsername must be a Minecraft username/,
  );
});

test('the reset command and read-back templates follow the pad anchor', () => {
  const bots = new ArenaBots(
    { padIndex: 3, padOriginX: 512, padOriginZ: 1024, learnerUsername: 'learner_3', dummyUsername: 'dummy_3' },
    { transport: { send: () => {} } },
  );

  assert.equal(
    bots._resetPadCommand(4),
    '/function arena:reset_pad {x:512,z:1024,learner:"learner_3",dummy:"dummy_3",nonce:4}',
  );
  // Feet at anchor+0.5; the dummy +3 further along x — exactly what
  // spawn_learner_pad / spawn_dummy_pad place.
  assert.deepEqual(bots.resetTemplate.position, { x: 512.5, y: 64.0, z: 1024.5 });
  assert.deepEqual(bots.dummyResetTemplate.position, { x: 515.5, y: 64.0, z: 1024.5 });
});

// ===========================================================================
// THE EXHIBITION RESET (T3). `arena:reset_pad`'s third line is
// `$function arena:spawn_dummy_pad {...dummy:"$(dummy)"...}`, and every one of
// THAT file's eleven lines addresses `$(dummy)` as a selector. With no dummy bot
// connected all eleven fail to find a player, so a human-mode reset prints
// eleven "no player found" errors into the console the operator is watching —
// per reset, all demo long. They are runtime selector no-ops rather than a macro
// abort (the `$`-substitutions stay syntactically valid, so the function does
// instantiate and its other lines run), which is why nothing was broken by them,
// only obscured.
//
// The key cannot simply be dropped — a macro function errors when a referenced
// key is absent, and THAT is the silent whole-function abort — so human mode
// issues the two lines of reset_pad that apply to it, directly. Which means the
// bridge now carries a COPY of datapack text: pinned below against the committed
// file so an edit there fails here, not during an exhibition.
// ===========================================================================

/** The non-comment lines of `reset_pad.mcfunction`, with `$(k)` substituted. */
function resetPadBody(args) {
  return datapackLines('reset_pad.mcfunction').map((line) =>
    line
      .replace(/^\$/, '')
      .replace(/\$\((\w+)\)/g, (_, key) => {
        assert.ok(key in args, `reset_pad references an unknown macro key $(${key})`);
        return String(args[key]);
      }),
  );
}

test('a human-mode reset runs the pad sweep and the LEARNER half, with no dummy selector at all', () => {
  const args = { x: 512, z: 1024, learner: 'learner_3', dummy: 'dummy_3', nonce: 7 };
  const body = resetPadBody(args);
  const commands = formatHumanResetCommands(args);

  // Line-for-line identity with what `arena:reset_pad` would have run, minus
  // its dummy call. A drift in either direction (the datapack changing its
  // sweep radius, this formatter changing its wording) fails here.
  assert.equal(commands.length, 2);
  assert.equal(commands[0], `/${body[0]}`, 'the entity sweep is the datapack line verbatim');
  assert.equal(
    commands[0],
    '/execute positioned 512 64 1024 run kill @e[type=!minecraft:player,distance=..64]',
  );
  assert.equal(commands[1], `/${body[1]}`, 'and the learner half is reset_pad\'s own call');
  assert.equal(
    commands[1],
    '/function arena:spawn_learner_pad {x:512,z:1024,learner:"learner_3",nonce:7}',
  );
  // The dummy half is what is deliberately NOT issued.
  assert.ok(body[2].startsWith('function arena:spawn_dummy_pad'), 'reset_pad calls it');
  assert.equal(
    commands.some((cmd) => cmd.includes('dummy')),
    false,
    'no exhibition command may name a bot that is not connected',
  );
  // The macro arguments are validated exactly as the reset_pad ones are: a bad
  // anchor here would still be pasted textually into spawn_learner_pad.
  assert.throws(
    () => formatHumanResetCommands({ ...args, x: -1 }),
    /pad anchor x must be a non-negative plain integer/,
  );
  assert.throws(
    () => formatHumanResetCommands({ ...args, learner: 'bad name' }),
    /learner username must be a Minecraft username/,
  );
  assert.throws(() => formatHumanResetCommands({ ...args, nonce: 1.5 }), /reset nonce/);
});

test('handleReset chats the dummy-free pair in human mode and the SINGLE macro in bot mode', async () => {
  // BOT MODE IS BYTE-INERT. The training path must issue exactly the one
  // command it always has — same string, same count — or M2 regresses.
  const botArena = new ArenaBots({}, { transport: { send: () => {} } });
  botArena.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  botArena.dummy = mockDummy();
  await botArena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  assert.deepEqual(botArena.learner.chatLog, [botArena._resetPadCommand(1)]);
  assert.deepEqual(botArena._resetCommands(1), [botArena._resetPadCommand(1)]);

  // HUMAN MODE: the same reset, with nothing addressed to an absent dummy.
  const sent = [];
  const humanArena = exhibitionArena(sent, {});
  humanArena._readbackOptions = SINGLE_POLL_GATE;
  humanArena.learner.chatLog = [];
  humanArena.learner.chat = (cmd) => humanArena.learner.chatLog.push(cmd);
  await humanArena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  assert.deepEqual(
    humanArena.learner.chatLog,
    formatHumanResetCommands({ x: 0, z: 0, learner: 'learner_bot', nonce: 1 }),
  );
});

test('the exhibition challenger is NOT logged as cross-pad contamination', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = exhibitionArena(sent, { 11: challenger });
  arena._readbackOptions = SINGLE_POLL_GATE;
  const errors = [];
  const realError = console.error;
  console.error = (...args) => errors.push(args.join(' '));
  try {
    // Claim the slot, then reset the way an operator does between challengers.
    await arena.handleStep({ type: 'step', action: Macro.IDLE });
    assert.equal(arena._claimedChallenger, 'classmate_1');
    assert.deepEqual(arena._scanForeignPlayers(), [], 'the opponent is not a foreign player');

    // DRIVEN THROUGH THE REAL handleReset, not just the scan, because the
    // ORDER of two lines at the end of it decides the outcome. handleReset both
    // scans AND releases the slot; releasing first would blank the exclusion
    // the scan depends on and log the outgoing challenger as contamination on
    // their way out — every reset, which is the whole bug. A direct
    // _scanForeignPlayers() call cannot see that ordering at all.
    await arena.handleReset({ type: 'reset', episode: 1, seed: 0 });
    assert.equal(arena._claimedChallenger, null, 'the reset still released the slot');

    // A SECOND person in the pad still IS foreign — that is the
    // one-challenger-at-a-time evidence, and narrowing the exclusion any
    // further would throw it away.
    await arena.handleStep({ type: 'step', action: Macro.IDLE });
    arena.learner.entities[22] = playerEntity('classmate_2', 8.5, 4.5);
    assert.deepEqual(arena._scanForeignPlayers(), ['classmate_2']);
  } finally {
    console.error = realError;
  }
  // eval/benchmark.py greps this exact line as contamination evidence, so an
  // exhibition that logged its own opponent would read as a compromised run.
  assert.equal(
    errors.filter((line) => line.includes('foreign_players') && line.includes('classmate_1'))
      .length,
    0,
    'the claimed challenger must never appear on the foreign_players line',
  );
  assert.ok(errors.some((line) => line.includes('foreign_players classmate_2')));
});

test('N=1 defaults reproduce the single-arena ports, usernames, anchor and templates (AC11)', () => {
  const bots = new ArenaBots({}, { transport: { send: () => {} } });

  assert.equal(bots.config.port, 25565);
  assert.equal(bots.config.bridgePort, 5555);
  assert.equal(bots.config.learnerUsername, 'learner_bot');
  assert.equal(bots.config.dummyUsername, 'dummy_bot');
  assert.deepEqual({ ...bots.padOrigin }, { x: 0, z: 0 });
  assert.equal(bots.padIndex, 0);
  // The literal template this file has asserted since before the pad topology.
  assert.deepEqual(bots.resetTemplate.position, { x: 0.5, y: 64.0, z: 0.5 });
  assert.deepEqual(bots.resetTemplate.inventory, ['iron_sword']);
  assert.equal(bots.resetTemplate.health, MAX_HEALTH);
  assert.deepEqual(bots.dummyResetTemplate.position, { x: 3.5, y: 64.0, z: 0.5 });
  assert.equal(formatSetupPadCommand(bots.padOrigin), '/function arena:setup_pad {x:0,z:0}');
});

test('handleReset issues ONE reset command, acks ok:true, then sends the initial state', async () => {
  const sent = [];
  const bots = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  // BOTH bots already read back their templates (live, the macro's /tp,
  // /effect and /give land during the gates' poll window; mocks have no async
  // command latency). The dummy gate is as load-bearing as the learner's:
  // acking while the dummy is still hurt would measure the first real hit
  // against a phantom baseline, so its mock must sit at the +3 x spawn, healed
  // and empty-handed.
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockDummy();

  await bots.handleReset({ type: 'reset', episode: 0, seed: 0 });

  // The frozen reset reply is TWO messages: reset_ack, then the post-reset
  // first observation the env's reset() blocks on (the live 30 s stall).
  assert.equal(sent.length, 2);
  assert.equal(sent[0].type, 'reset_ack');
  assert.equal(sent[0].ok, true);
  assert.deepEqual(sent[0].readback.inventory, ['iron_sword']);
  assert.equal(sent[1].type, 'state');
  // Schema-valid on the wire, with a clean event window and the reset tick.
  assert.doesNotThrow(() => validateOutbound(sent[1]));
  assert.equal(sent[1].tick, 0);
  assert.deepEqual(sent[1].events, {
    damage_dealt: 0,
    damage_taken: 0,
    i_died: false,
    opponent_died: false,
  });
  // EXACTLY ONE command, and it is the pad's reset macro — the whole point of
  // the single-authority design. A second bridge-side command here (the old
  // unconditional `/effect clear` in particular) could land in the same tick
  // after the datapack's instant heal + saturation and strip them, silently
  // voiding the food restore AC18 depends on. The dummy is never chatted at:
  // one macro call resets both bots.
  assert.deepEqual(bots.learner.chatLog, [
    '/function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot",nonce:1}',
  ]);
  assert.deepEqual(bots.dummy.chatLog, [], 'the reset is one command from one bot');
});

// ===========================================================================
// PER-OPPONENT MOBILITY OVERRIDE (T11c). spawn_dummy_pad.mcfunction always
// pins knockback_resistance to 1.0 AND movement_speed to 0.0 — right for the
// M2 stationary dummy, wrong for a scripted opponent
// (OpponentConfig.knockback_immune=False), which must be knockable AND able to
// walk. These pin the bridge-side override handleReset issues instead of a new
// macro key, and the read-back that proves the server accepted it.
// ===========================================================================

test('dummyKnockbackImmune defaults to true and handleReset then sends the dummy NOTHING (M2 byte-identical)', async () => {
  const sent = [];
  const bots = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  assert.equal(bots.dummyKnockbackImmune, true, 'the default must reproduce M2 exactly');
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockDummy();

  await bots.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent[0].ok, true);
  assert.deepEqual(
    bots.learner.chatLog,
    ['/function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot",nonce:1}'],
    'still exactly the one reset macro',
  );
  assert.deepEqual(
    bots.dummy.chatLog,
    [],
    'a true (immune) opponent needs no override: the datapack\'s own 1.0 already applies',
  );
});

test('dummyKnockbackImmune=false makes handleReset un-pin BOTH attributes, via the DUMMY\'s own connection only (AC18)', async () => {
  const sent = [];
  const bots = new ArenaBots(
    { dummyKnockbackImmune: false },
    { transport: { send: (msg) => sent.push(msg) } },
  );
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockDummy();

  await bots.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent[0].ok, true);
  // The learner's half is UNCHANGED — still the single reset_pad macro, same
  // string, same count. A regression here would mean the toggle leaked into
  // the byte-inert bot-mode path the plan explicitly protects.
  assert.deepEqual(bots.learner.chatLog, [
    '/function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot",nonce:1}',
  ]);
  // BOTH overrides ride the DUMMY's own connection, addressed to itself — never
  // the learner's — so neither can be mistaken for a second reset-authority
  // command. The movement half is NOT optional: the datapack pins the opponent
  // to speed 0.0, and a ScriptedBot whose APPROACH/STRAFE/RETREAT do nothing is
  // a stationary target wearing a scripted opponent's name.
  //
  // No `base get` follows here: this fake models no chat channel (mockDummy has
  // no `on`), so no reply could ever arrive and the read-back is skipped as
  // vacuous — the same tolerance _resetWasConfirmed extends to the same fakes.
  assert.deepEqual(bots.dummy.chatLog, [
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base set ${KNOCKBACK_RESISTANCE_NOT_IMMUNE_VALUE}`,
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base set ${MOVEMENT_SPEED_MOBILE_VALUE}`,
  ]);
});

test('dummyKnockbackImmune=false still sends NO override when the datapack reset was never confirmed', async () => {
  // Exactly the silent-abort trap the rest of this suite already covers for
  // the reset ack itself: both gates match (the post-kill posture looks like
  // a fresh reset) but no beacon arrives, so `confirmed` is false. Overriding
  // here would assume the datapack's own `base set 1.0` already ran — which,
  // in this scenario, it did not (the whole function aborted at instantiation)
  // — and the override could be composing on top of an unknown prior state.
  const sent = [];
  const errors = [];
  const arena = new ArenaBots(
    { dummyKnockbackImmune: false },
    {
      transport: { send: (msg) => sent.push(msg) },
      readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
    },
  );
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  const realError = console.error;
  console.error = (msg) => errors.push(String(msg));
  try {
    await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  } finally {
    console.error = realError;
  }

  assert.equal(sent[0].ok, false, 'a state that merely LOOKS reset is not a reset');
  assert.deepEqual(
    arena.dummy.chatLog,
    [],
    'no override without proof the datapack (and its base set 1.0) actually ran',
  );
});

test('an ArenaBots built with a non-boolean dummyKnockbackImmune fails at construction', () => {
  for (const bad of ['false', 0, 1, null, 'true']) {
    assert.throws(
      () => new ArenaBots({ dummyKnockbackImmune: bad }, { transport: { send: () => {} } }),
      /dummyKnockbackImmune must be a boolean/,
      `dummyKnockbackImmune=${String(bad)} must be rejected`,
    );
  }
});

// --- the read-back: proof the SERVER accepted the override -----------------
// _sendCommand chats and returns, so on its own the override is exactly the
// fire-and-forget silent failure the datapack's beacon exists to eliminate. A
// bad attribute id after a version bump, or a dummy that is not opped, would
// leave the opponent immune and immobile with nothing in any log. These pin the
// `base get` round trip that closes it — and pin that it stays LOG-ONLY, since
// a reply-shape drift must never be able to abort a training run.

/** Run `body` with console.error captured. */
async function withCapturedErrors(body) {
  const errors = [];
  const realError = console.error;
  console.error = (msg) => errors.push(String(msg));
  try {
    await body();
  } finally {
    console.error = realError;
  }
  return errors;
}

/** A confirmed, non-immune reset on EventEmitter bots (the live-ish shape). */
function nonImmuneArena(sent, readbackOptions = {}) {
  const arena = new ArenaBots(
    { dummyKnockbackImmune: false },
    {
      transport: { send: (msg) => sent.push(msg) },
      readbackOptions: { timeoutMs: 0, pollIntervalMs: 1, ...readbackOptions },
    },
  );
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);
  return arena;
}

test('the override is READ BACK: each base set is followed by a base get on the same connection', async () => {
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 500 });
  answerAttributeGetsLikeTheServer(arena);

  const errors = await withCapturedErrors(() =>
    arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );

  assert.equal(sent[0].ok, true);
  // SET then GET, in that order, for BOTH attributes: the get must follow its
  // own set or it reads the datapack's value and confirms the wrong thing.
  assert.deepEqual(arena.dummy.chatLog, [
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base set ${KNOCKBACK_RESISTANCE_NOT_IMMUNE_VALUE}`,
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base set ${MOVEMENT_SPEED_MOBILE_VALUE}`,
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base get`,
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base get`,
  ]);
  // SILENCE IS THE CONFIRMATION, exactly like the rl_deaths read-back: a healthy
  // override says nothing at all, so the ABSENCE of these lines is what an
  // operator watching the console reads as "the opponent can be knocked back".
  assert.deepEqual(errors, [], 'a confirmed override is silent');
  // The window closes with the reset: a reply that arrives late must not be
  // credited to the next one.
  assert.equal(arena._attributeReadback, null);
});

test('a server that never answers the read-back is named LOUDLY, per attribute, without failing the reset', async () => {
  const sent = [];
  // attributeReadbackTimeoutMs: 0 models the reply never coming (a bad
  // attribute id after a version bump, an un-opped dummy, sendCommandFeedback
  // off) without making the test wait for a real timeout.
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 0 });

  const errors = await withCapturedErrors(() =>
    arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );

  assert.equal(sent[0].ok, true, 'LOG-ONLY: an unconfirmed override must not abort the run');
  for (const attribute of [KNOCKBACK_RESISTANCE_ATTRIBUTE, MOVEMENT_SPEED_ATTRIBUTE]) {
    assert.ok(
      errors.some((line) => line.includes(attribute) && line.includes('NOT confirmed')),
      `${attribute} must be named on stderr when the server stays silent`,
    );
  }
  assert.ok(
    errors.some((line) => line.includes('not opped')),
    'the log must name the causes, not just the symptom',
  );
});

test('one attribute confirming does not cover for the other', async () => {
  // The realistic partial failure: knockback_resistance is fine and
  // movement_speed is the one that broke (it is the id that changed name in
  // 1.21.2 and again in 1.21.4). A per-attribute report is what keeps that from
  // hiding behind an all-or-nothing gate.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 60 });
  answerAttributeGetsLikeTheServer(arena, { silentFor: [MOVEMENT_SPEED_ATTRIBUTE] });

  const errors = await withCapturedErrors(() =>
    arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );

  assert.ok(
    errors.some((line) => line.includes(MOVEMENT_SPEED_ATTRIBUTE) && line.includes('NOT confirmed')),
    'the silent attribute is reported',
  );
  assert.ok(
    !errors.some((line) => line.includes(KNOCKBACK_RESISTANCE_ATTRIBUTE)),
    'the attribute that DID confirm is not reported',
  );
});

test('a server that applies a DIFFERENT value than we set is reported as REJECTED', async () => {
  // The case a "did the command run?" check cannot see: the server answers, so
  // the round trip is healthy, but the value is not ours — e.g. something
  // re-pinned the dummy between the set and the get.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 500 });
  answerAttributeGetsLikeTheServer(arena, {
    values: { [KNOCKBACK_RESISTANCE_ATTRIBUTE]: Number(KNOCKBACK_RESISTANCE_IMMUNE_VALUE) },
  });

  const errors = await withCapturedErrors(() =>
    arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );

  assert.equal(sent[0].ok, true, 'still log-only');
  assert.ok(
    errors.some(
      (line) => line.includes(KNOCKBACK_RESISTANCE_ATTRIBUTE) && line.includes('REJECTED'),
    ),
    'a wrong value is a different diagnosis from a missing reply, and must read that way',
  );
  assert.ok(
    !errors.some((line) => line.includes(MOVEMENT_SPEED_ATTRIBUTE)),
    'the attribute that landed correctly is not reported',
  );
});

test('the read-back accepts the value as a JSON NUMBER, which is how Paper sends it', () => {
  // `base set 0.0` comes back through JSON.parse as the JS number 0, so a
  // TEXTUAL comparison against the "0.0" we sent would flag every healthy
  // override as rejected. Driven through the real ChatMessage so the test
  // cannot agree with the code by construction.
  const arena = new ArenaBots({ dummyKnockbackImmune: false }, { transport: { send: () => {} } });
  arena._attributeReadback = new Map();
  arena._captureAttributeReadback(
    new ChatMessage({
      translate: ATTRIBUTE_GET_TRANSLATE_KEY,
      with: [{ translate: KNOCKBACK_RESISTANCE_NAME_KEY }, { text: 'dummy_bot' }, 0.0],
    }),
  );
  assert.equal(arena._attributeReadback.get(KNOCKBACK_RESISTANCE_NAME_KEY), 0);
});

test('the read-back ignores every message that is not an attribute reply', () => {
  const arena = new ArenaBots({ dummyKnockbackImmune: false }, { transport: { send: () => {} } });
  arena._attributeReadback = new Map();
  const ignored = [
    null,
    undefined,
    'a plain string',
    new ChatMessage({ text: '[arena] reset_ok dummy 0 0 dummy_bot 1' }),
    // Right key, an attribute we never set.
    new ChatMessage({
      translate: ATTRIBUTE_GET_TRANSLATE_KEY,
      with: [{ translate: 'attribute.name.generic.max_health' }, { text: 'dummy_bot' }, 20],
    }),
    // Right key, truncated args.
    new ChatMessage({
      translate: ATTRIBUTE_GET_TRANSLATE_KEY,
      with: [{ translate: KNOCKBACK_RESISTANCE_NAME_KEY }],
    }),
  ];
  for (const message of ignored) {
    arena._captureAttributeReadback(message);
  }
  assert.equal(arena._attributeReadback.size, 0);
});

test('a reset superseded DURING the read-back wait does not ack', async () => {
  // The read-back adds an await point AFTER the two epoch guards inside
  // handleReset's try, so it needs its own. Without it a stale handler would
  // ack a reset the retry already owns and desync the request/reply stream —
  // the exact failure the other two guards exist to prevent.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 60 });
  answerAttributeGetsLikeTheServer(arena, { silentFor: [MOVEMENT_SPEED_ATTRIBUTE] });
  // TRIGGERED BY THE READ-BACK ITSELF, not by a timer. A timer racing the
  // causality wait would trip the epoch guard that already exists INSIDE the
  // try, and the test would pass without the new guard ever running — it did,
  // until a mutation run caught it. The `base get` is only ever chatted after
  // `confirmed` is true and the two earlier guards are behind us, so hooking it
  // puts the retry exactly where it must be: between asking and reading.
  const askAgain = arena.dummy.chat;
  arena.dummy.chat = (cmd) => {
    askAgain(cmd);
    if (typeof cmd === 'string' && cmd.endsWith('base get')) {
      arena._resetEpoch += 1; // what a retry's `++this._resetEpoch` does
    }
  };

  await withCapturedErrors(() => arena.handleReset({ type: 'reset', episode: 0, seed: 0 }));

  assert.ok(
    arena.dummy.chatLog.some((cmd) => cmd.endsWith('base get')),
    'the read-back must actually have been reached, or this proves nothing',
  );
  assert.deepEqual(sent, [], 'a stale handler acks nothing at all');
});

test('a dummy that loses its connection mid-override strands no read-back window', async () => {
  // The override adds a new place where the DUMMY's chat() runs, and this repo
  // has been bitten before by a flag raised on one path and stranded on
  // another. A throw here propagates exactly as the learner's does (see
  // "handleReset rejects on a throwing chat()"), and must leave
  // _attributeReadback closed so a stray reply cannot be credited to the next
  // reset.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 60 });
  // Throw on the READ-BACK specifically, not on the `base set` above it: the
  // window is armed by then, so this is the only arrangement that exercises the
  // finally that closes it.
  const realChat = arena.dummy.chat;
  arena.dummy.chat = (cmd) => {
    if (typeof cmd === 'string' && cmd.endsWith('base get')) {
      throw new Error('cannot chat: bot is not spawned');
    }
    realChat(cmd);
  };

  await assert.rejects(
    () => withCapturedErrors(() => arena.handleReset({ type: 'reset', episode: 0, seed: 0 })),
    /cannot chat/,
  );

  assert.equal(arena._attributeReadback, null, 'the capture window is not left open');
  assert.equal(arena._suppressOpponentEvents, false, 'and suppression is not stranded either');
  assert.deepEqual(sent, [], 'a reset that threw must not ack');
});

test('a stale read-back wait does NOT close the window a retry already opened', async () => {
  // Reset is reconnect-and-retry, so a retry can run its whole gate sequence
  // and arm its own capture window while this handler is still parked. If the
  // stale handler's cleanup nulled the window unconditionally, the RETRY's
  // replies would land on a closed window and it would log two false "NOT
  // confirmed" lines — false alarms on the exact line RUNBOOK Step 2c tells the
  // operator to trust, which is worse than no read-back at all.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 30 });
  // A retry arming its own window, timed off this handler's own `base get` so
  // it lands strictly inside the wait (a timer would race the causality gate).
  const retryWindow = new Map();
  const realChat = arena.dummy.chat;
  arena.dummy.chat = (cmd) => {
    realChat(cmd);
    if (typeof cmd === 'string' && cmd.endsWith('base get')) {
      arena._attributeReadback = retryWindow;
      arena._resetEpoch += 1; // the retry owns the episode now
    }
  };

  await withCapturedErrors(() => arena.handleReset({ type: 'reset', episode: 0, seed: 0 }));

  assert.equal(
    arena._attributeReadback,
    retryWindow,
    "the stale handler must not close a window it did not open",
  );
});

test('a superseded read-back wait logs NOTHING, even though its own `seen` never fills', async () => {
  // Same supersession as the test above, but pinning the OTHER half of the
  // contract: the identity-guarded finally keeps the stale handler from
  // closing the retry's window, but that alone does not stop the stale
  // handler from walking its own (permanently empty) `seen` map afterward and
  // printing two false "NOT confirmed" lines — on the exact line RUNBOOK Step
  // 2c tells the operator to trust as the confirmation. The prior test does
  // not capture console.error, so it passes whether or not those false lines
  // print; this one binds it.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 30 });
  const retryWindow = new Map();
  const realChat = arena.dummy.chat;
  arena.dummy.chat = (cmd) => {
    realChat(cmd);
    if (typeof cmd === 'string' && cmd.endsWith('base get')) {
      arena._attributeReadback = retryWindow;
      arena._resetEpoch += 1; // the retry owns the episode now
    }
  };

  const errors = await withCapturedErrors(() =>
    arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );

  assert.ok(
    arena.dummy.chatLog.some((cmd) => cmd.endsWith('base get')),
    'the read-back must actually have been reached, or this proves nothing',
  );
  assert.deepEqual(
    errors,
    [],
    'a superseded handler must log nothing — the retry, not this handler, speaks for this reset',
  );
});

test('the read-back rides the beacon channel WITHOUT disturbing it', async () => {
  // Both live on the dummy's `message` events. The beacon is matched on exact
  // text and the read-back structurally, so neither can consume the other's
  // message — and the beacon is what gates the override in the first place.
  const sent = [];
  const arena = nonImmuneArena(sent, { attributeReadbackTimeoutMs: 500 });
  answerAttributeGetsLikeTheServer(arena);

  await withCapturedErrors(() => arena.handleReset({ type: 'reset', episode: 0, seed: 0 }));

  assert.deepEqual(arena._resetConfirm, { nonce: 1, learner: true, dummy: true });
});

// ===========================================================================
// RESET CAUSALITY (the gate verifies template MATCH, not that the reset ran).
//
// After a kill cycle the natural post-respawn state IS the template state: the
// dummy respawns at its pinned spawnpoint at full health, empty-handed, with
// effects cleared by death, and a learner that killed from its spawn without
// moving still reads back 20 / anchor+0.5 / ['iron_sword'] / no effects. So a
// reset_pad that ABORTS AT INSTANTIATION would pass both gates and ack a reset
// that never happened — no saturation restore, no knockback re-pin, silently,
// and precisely under the combat probe's stationary kill cycles.
//
// The datapack ends each spawn function with a beacon a bare respawn cannot
// produce. These tests pin both directions.
// ===========================================================================

test('handleReset acks ok:FALSE when both gates match but the datapack never confirmed', async () => {
  const sent = [];
  const errors = [];
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    // Both bots match on the first poll, so the gates consume none of their
    // budget; the confirmation wait then runs on its MIN_CONFIRM_WAIT_MS floor
    // (250 ms) instead of the full 3 s envelope, keeping the suite fast while
    // still exercising a real bounded wait.
    readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
  });
  // Exactly the post-kill trap: both bots ALREADY look reset. No
  // answerResetLikeTheServer() here — the macro aborted, so no beacon arrives.
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  const realError = console.error;
  console.error = (msg) => errors.push(String(msg));
  try {
    await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  } finally {
    console.error = realError;
  }

  assert.equal(sent.length, 1, 'no first observation: the episode must not start');
  assert.equal(sent[0].type, 'reset_ack');
  assert.equal(sent[0].ok, false, 'a state that merely LOOKS reset is not a reset');
  assert.ok(
    errors.some((line) => line.includes('reset NOT confirmed by the datapack')),
    'the silent case is named loudly on stderr',
  );
});

test('handleReset acks ok:true once BOTH beacons arrive, and re-arms the latch each reset', async () => {
  const sent = [];
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
  });
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  assert.equal(sent.length, 2);
  assert.equal(sent[0].ok, true);
  assert.deepEqual(arena._resetConfirm, { nonce: 1, learner: true, dummy: true });

  // The latch is per-reset, not sticky: a later reset whose macro aborts must
  // NOT inherit this one's confirmation.
  arena.learner.chat = () => {};
  await arena.handleReset({ type: 'reset', episode: 1, seed: 1 });
  assert.equal(sent.length, 3, 'ack only');
  assert.equal(sent[2].ok, false, 'the latch was re-armed, so the silent reset is caught');
});

test('the confirmation is WAITED for: a beacon that lands after the gates still acks ok:true', async () => {
  // THE REGRESSION THIS PINS. runReadbackGate snapshots synchronously on entry
  // and returns {ok:true} from its first iteration with no await, so when both
  // bots already match — the post-kill posture, i.e. every reset of the combat
  // probe — handleReset runs from the reset command straight to the confirm
  // check without ever yielding to the macrotask queue. The beacon CANNOT have
  // arrived: the command has not even reached the server. Checking the latch
  // without waiting therefore failed every healthy reset, and because the
  // retry's gates match at t=0 too, the env's second attempt failed the same
  // way and raised. A 200 ms beacon (far beyond any real round trip) proves the
  // wait is real and bounded by the gate budget, not by luck.
  const sent = [];
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena, { delayMs: 200 });

  const startedAt = Date.now();
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent.length, 2, 'ack + first observation');
  assert.equal(sent[0].ok, true);
  assert.ok(Date.now() - startedAt >= 150, 'the reset really waited for the round trip');
});

test('a LATE beacon from the previous reset cannot confirm the next one (nonce)', async () => {
  const sent = [];
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
  });
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();

  // Reset 1's beacons, arriving now — long after reset 1 gave up. Without the
  // nonce these texts are indistinguishable from reset 2's and would confirm a
  // reset that never ran; the beacon is the last line of the same function, so
  // "a beacon exists" must mean "THIS reset's function ran".
  const stale = {
    learner: arena._resetConfirmationText('learner', 1),
    dummy: arena._resetConfirmationText('dummy', 1),
  };
  arena.learner.chat = () => {
    arena.learner.emit('message', stale.learner);
    arena.dummy.emit('message', stale.dummy);
  };

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 }); // epoch/nonce 1
  await arena.handleReset({ type: 'reset', episode: 1, seed: 1 }); // epoch/nonce 2

  assert.equal(sent.length, 3, 'reset 1 acked + observed; reset 2 acked only');
  assert.equal(sent[0].ok, true, 'reset 1 IS confirmed by its own beacon');
  assert.equal(sent[2].ok, false, "reset 2 is not confirmed by reset 1's beacon");
});

test('only THIS pad\'s beacon confirms it — a neighbour\'s cannot', async () => {
  const sent = [];
  const arena = new ArenaBots(
    { padIndex: 3, padOriginX: 512, padOriginZ: 0, learnerUsername: 'learner_3', dummyUsername: 'dummy_3' },
    { transport: { send: (msg) => sent.push(msg) }, readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 } },
  );
  arena.learner = liveBot('learner_3', { position: { x: 512.5, y: 64, z: 0.5 } });
  arena.dummy = liveBot('dummy_3', { inventory: [], position: { x: 515.5, y: 64, z: 0.5 } });
  arena.wireDamageEvents();
  // Pad 4's beacons, and pad 3's own text with the roles swapped: neither may
  // confirm this pad. (The datapack addresses beacons by name, so this is
  // defense in depth — but a cross-pad confirmation would re-create the exact
  // attribution hazard walls and spacing exist to prevent.)
  arena.learner.chat = () => {
    arena.learner.emit('message', '[arena] reset_ok learner 1024 0 learner_4');
    arena.dummy.emit('message', '[arena] reset_ok dummy 512 0 dummy_4');
    arena.dummy.emit('message', arena._resetConfirmationText('learner'));
  };

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent[0].ok, false);
  assert.deepEqual(arena._resetConfirm, { nonce: 1, learner: false, dummy: false });
  assert.equal(
    arena._resetConfirmationText('learner'),
    '[arena] reset_ok learner 512 0 learner_3 1',
    'the beacon carries the anchor, the username AND the per-reset nonce',
  );
});

// ===========================================================================
// STATIC DATAPACK CONTRACTS. These read the committed .mcfunction files and
// pin the two orderings the bridge now depends on but no longer performs. They
// replace the executable coverage the deleted `_regear` tests used to give the
// clear-before-give rule, which is load-bearing: the instant_health and
// saturation instances live for ONE gametick, so a trailing `effect clear` in
// the same tick would strip them before they ever applied — silently voiding
// the food restore AC18 rides on.
// ===========================================================================

/** Read one committed arena function as an array of non-comment lines. */
function datapackLines(name) {
  const file = path.join(__dirname, '..', 'server', 'arena', 'data', 'arena', 'function', name);
  return fs
    .readFileSync(file, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'));
}

test('spawn_learner_pad clears BEFORE it gives, with no trailing clear', () => {
  const lines = datapackLines('spawn_learner_pad.mcfunction');
  const effectClear = lines.findIndex((line) => line.startsWith('$effect clear'));
  const lastEffectGive = lines.map((line) => line.startsWith('$effect give')).lastIndexOf(true);
  const invClear = lines.findIndex((line) => line.startsWith('$clear '));
  const give = lines.findIndex((line) => line.startsWith('$give '));

  assert.ok(effectClear >= 0 && lastEffectGive >= 0, 'the file clears and grants effects');
  assert.ok(effectClear < lastEffectGive, '$effect clear must precede every $effect give');
  assert.equal(
    lines.slice(lastEffectGive).filter((line) => line.startsWith('$effect clear')).length,
    0,
    'a trailing $effect clear would strip the one-gametick instant effects',
  );
  assert.ok(invClear >= 0 && give > invClear, '$clear must precede the regear $give');
  assert.ok(
    lines[lines.length - 1].startsWith('$tellraw $(learner)'),
    'the causality beacon must stay the LAST line',
  );
});

test('spawn_dummy_pad clears before it gives, gives the dummy NO weapon, and ends with its beacon', () => {
  const lines = datapackLines('spawn_dummy_pad.mcfunction');
  const effectClear = lines.findIndex((line) => line.startsWith('$effect clear'));
  const lastEffectGive = lines.map((line) => line.startsWith('$effect give')).lastIndexOf(true);

  assert.ok(effectClear >= 0 && lastEffectGive >= 0);
  assert.ok(effectClear < lastEffectGive, '$effect clear must precede every $effect give');
  assert.equal(
    lines.slice(lastEffectGive).filter((line) => line.startsWith('$effect clear')).length,
    0,
    'a trailing $effect clear would silently void the dummy heal and food restore',
  );
  // The other half of the bridge/datapack template agreement: nothing arms the
  // dummy, which is why dummyResetTemplate.inventory is [].
  assert.equal(
    lines.filter((line) => line.startsWith('$give ')).length,
    0,
    'the dummy is a passive target: no weapon, ever',
  );
  assert.ok(lines.some((line) => line.startsWith('$clear $(dummy)')), 'the dummy inventory is cleared');
  assert.ok(
    lines[lines.length - 1].startsWith('$tellraw $(dummy)'),
    'the causality beacon must stay the LAST line',
  );
});

test('the datapack\'s knockback attribute id and default value are pinned against bot.js (T11c)', () => {
  // spawn_dummy_pad.mcfunction stays the single source of truth for what
  // "immune" means; bot.js's override composes its OWN command from module
  // constants rather than re-reading this file live, so a drift between the
  // two — e.g. someone flattening the `generic.` infix here without touching
  // bot.js — must fail HERE, in CI, not during a live exhibition where the
  // override would silently target an attribute id the server does not have.
  const lines = datapackLines('spawn_dummy_pad.mcfunction');
  const kbLine = lines.find((line) => line.includes('knockback_resistance'));
  assert.ok(kbLine, 'spawn_dummy_pad.mcfunction must set knockback_resistance somewhere');
  const parsed = /^\$attribute \$\(dummy\) (\S+) base set (\S+)$/.exec(kbLine);
  assert.ok(parsed, `unexpected knockback attribute line shape: ${kbLine}`);
  const [, attributeId, value] = parsed;
  assert.equal(
    attributeId,
    KNOCKBACK_RESISTANCE_ATTRIBUTE,
    "bot.js's override must address the SAME attribute id this file applies",
  );
  assert.equal(
    value,
    KNOCKBACK_RESISTANCE_IMMUNE_VALUE,
    'bot.js\'s notion of "what immune means" must match what this file actually sets',
  );
});

test('the datapack\'s movement_speed attribute id and pinned value match bot.js (T11c)', () => {
  // The knockback twin, and the more load-bearing of the two: this is the line
  // whose value the plan calls "a belt-and-suspenders anti-drift measure" and
  // which pins a scripted opponent to zero speed. If someone flattens the
  // `generic.` infix here, or changes the pinned 0.0, the bridge's override
  // would target an id the server does not have — and APPROACH/STRAFE/RETREAT
  // would go inert with a completely clean log. Fail HERE instead.
  const lines = datapackLines('spawn_dummy_pad.mcfunction');
  const speedLine = lines.find((line) => line.includes('movement_speed'));
  assert.ok(speedLine, 'spawn_dummy_pad.mcfunction must set movement_speed somewhere');
  const parsed = /^\$attribute \$\(dummy\) (\S+) base set (\S+)$/.exec(speedLine);
  assert.ok(parsed, `unexpected movement_speed attribute line shape: ${speedLine}`);
  const [, attributeId, value] = parsed;
  assert.equal(
    attributeId,
    MOVEMENT_SPEED_ATTRIBUTE,
    "bot.js's override must address the SAME attribute id this file applies",
  );
  assert.equal(
    value,
    MOVEMENT_SPEED_STATIONARY_VALUE,
    'bot.js\'s notion of "what stationary means" must match what this file actually sets',
  );
});

test('formatKnockbackResistanceCommand renders the immune and non-immune attribute commands', () => {
  assert.equal(
    formatKnockbackResistanceCommand('dummy_bot', true),
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base set ${KNOCKBACK_RESISTANCE_IMMUNE_VALUE}`,
  );
  assert.equal(
    formatKnockbackResistanceCommand('dummy_bot', false),
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base set ${KNOCKBACK_RESISTANCE_NOT_IMMUNE_VALUE}`,
  );
});

test('formatKnockbackResistanceCommand rejects a bad username or a non-boolean immune flag', () => {
  assert.throws(
    () => formatKnockbackResistanceCommand('bad name', false),
    /dummy username must be a Minecraft username/,
  );
  for (const bad of ['false', 0, null, undefined]) {
    assert.throws(
      () => formatKnockbackResistanceCommand('dummy_bot', bad),
      /knockback immune flag must be a boolean/,
      `immune=${String(bad)} must be rejected`,
    );
  }
});

test('formatMovementSpeedCommand renders the stationary and mobile attribute commands', () => {
  assert.equal(
    formatMovementSpeedCommand('dummy_bot', true),
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base set ${MOVEMENT_SPEED_STATIONARY_VALUE}`,
  );
  assert.equal(
    formatMovementSpeedCommand('dummy_bot', false),
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base set ${MOVEMENT_SPEED_MOBILE_VALUE}`,
  );
  // 0.1 is a vanilla PLAYER's walking speed, not a round number someone liked:
  // Player.createAttributes() in the pinned jar adds MOVEMENT_SPEED at
  // 0.10000000149011612d, and prismarine-physics uses the same 0.1 fallback.
  assert.equal(Number(MOVEMENT_SPEED_MOBILE_VALUE), 0.1);
});

test('formatMovementSpeedCommand rejects a bad username or a non-boolean stationary flag', () => {
  assert.throws(
    () => formatMovementSpeedCommand('bad name', false),
    /dummy username must be a Minecraft username/,
  );
  for (const bad of ['false', 0, null, undefined]) {
    assert.throws(
      () => formatMovementSpeedCommand('dummy_bot', bad),
      /movement stationary flag must be a boolean/,
      `stationary=${String(bad)} must be rejected`,
    );
  }
});

test('formatAttributeGetCommand renders a read-back only for the two attributes the override writes', () => {
  assert.equal(
    formatAttributeGetCommand('dummy_bot', KNOCKBACK_RESISTANCE_ATTRIBUTE),
    `/attribute dummy_bot ${KNOCKBACK_RESISTANCE_ATTRIBUTE} base get`,
  );
  assert.equal(
    formatAttributeGetCommand('dummy_bot', MOVEMENT_SPEED_ATTRIBUTE),
    `/attribute dummy_bot ${MOVEMENT_SPEED_ATTRIBUTE} base get`,
  );
  // Not a general-purpose accessor: a typo'd id would produce a command whose
  // reply nothing is waiting for, which reads exactly like a server that never
  // answered.
  for (const bad of ['minecraft:generic.max_health', 'generic.movement_speed', '', null]) {
    assert.throws(
      () => formatAttributeGetCommand('dummy_bot', bad),
      /unsupported read-back attribute/,
      `attribute=${String(bad)} must be rejected`,
    );
  }
  assert.throws(
    () => formatAttributeGetCommand('bad name', MOVEMENT_SPEED_ATTRIBUTE),
    /dummy username must be a Minecraft username/,
  );
});

/** The generated world copy Paper actually reads (gitignored; absent on a fresh clone). */
const LIVE_DATAPACK_DIR = path.join(
  __dirname,
  '..',
  'server',
  'world',
  'datapacks',
  'arena',
  'data',
  'arena',
  'function',
);

test('the committed datapack matches the one the server actually loads', { skip: !fs.existsSync(LIVE_DATAPACK_DIR) }, () => {
  // server/world/ is generated and gitignored, so this only runs on a machine
  // that has actually booted the server — where an edit landing in only one of
  // the two copies means Paper is loading a stale datapack.
  for (const name of [
    'reset.mcfunction',
    'reset_pad.mcfunction',
    'setup_pad.mcfunction',
    'spawn_learner_pad.mcfunction',
    'spawn_dummy_pad.mcfunction',
  ]) {
    const src = path.join(__dirname, '..', 'server', 'arena', 'data', 'arena', 'function', name);
    assert.equal(
      fs.readFileSync(path.join(LIVE_DATAPACK_DIR, name), 'utf8'),
      fs.readFileSync(src, 'utf8'),
      `${name} is out of sync — re-copy server/arena into server/world/datapacks/arena`,
    );
  }
});

test('_scanForeignPlayers names only players that are not this pad\'s two bots (AC13 evidence)', () => {
  const bots = new ArenaBots(
    { padIndex: 3, padOriginX: 512, padOriginZ: 0, learnerUsername: 'learner_3', dummyUsername: 'dummy_3' },
    { transport: { send: () => {} } },
  );
  // `dummy.on('health')` records a health DROP with NO attacker attribution, so
  // a bot that reached a neighbouring pad would silently credit its damage to
  // that pad's policy. This scan is the observable that proves it never
  // happens; T12 consumes the log line, and it never touches the frozen wire.
  bots.learner = {
    username: 'learner_3',
    entities: {
      1: { type: 'player', username: 'learner_3' },
      2: { type: 'player', username: 'dummy_3' },
      3: { type: 'player', username: 'learner_4' },
      4: { type: 'player', username: 'learner_4' }, // duplicate: reported once
      5: { type: 'orb' },
      6: { type: 'player' }, // no username yet
      7: null,
    },
  };

  assert.deepEqual(bots._scanForeignPlayers(), ['learner_4']);

  bots.learner.entities = { 1: { type: 'player', username: 'dummy_3' } };
  assert.deepEqual(bots._scanForeignPlayers(), [], 'own bots are never foreign');

  bots.learner = null;
  assert.deepEqual(bots._scanForeignPlayers(), [], 'a bot with no entity view scans clean');
});

test('_updateLastSeen stores a Vec3-style clone, not an alias of the live position', () => {
  const bots = new ArenaBots({}, { transport: { send: () => {} } });
  const livePos = {
    x: 3.5,
    y: 64,
    z: 0.5,
    clone() {
      return { x: this.x, y: this.y, z: this.z, cloned: true };
    },
  };
  bots.dummy = { entity: { position: livePos } };

  bots._updateLastSeen();

  // The clone (with its Vec3 methods, live) is stored — never the live vector,
  // which keeps moving after the opponent was last seen.
  assert.notEqual(bots._lastSeenOpponentPos, livePos);
  assert.equal(bots._lastSeenOpponentPos.cloned, true);
  assert.deepEqual(
    { x: bots._lastSeenOpponentPos.x, y: bots._lastSeenOpponentPos.y, z: bots._lastSeenOpponentPos.z },
    { x: 3.5, y: 64, z: 0.5 },
  );
});

// ===========================================================================
// TC14 (AC11) — GUARD TEST, NOT A SPEC. Read this before touching either the
// test or _updateLastSeen.
//
// _updateLastSeen() writes the opponent's LIVE position UNCONDITIONALLY, every
// window, with no field-of-view or line-of-sight check at all — see its own
// TODO(T12) comment in bridge/bot.js. That is a KNOWN CONTRACT VIOLATION: the
// macro it feeds, TURN_TO_LAST_SEEN, is meant to face a position the agent
// genuinely SAW, and today it is an omniscient aim-snap instead. The Contracts
// table in docs/plans/2026-08-16-demo-scripted-opponent-exhibition.md FREEZES
// this exact unconditional write through 2026-08-20: gating it removes the
// agent's only way to re-acquire an opponent it cannot see, with no
// replacement shipped yet, which would make the demo worse, not more honest.
//
// This test therefore asserts the OPPOSITE of correct behavior on purpose. If
// it starts failing, the two live possibilities are:
//   1. Someone gated _updateLastSeen() before 2026-08-20 — revert that change,
//      it breaks the frozen contract and the demo with it.
//   2. It is after 2026-08-20 and TODO(T12) has been resolved with real
//      PerceptionFilter-backed gating — in which case DELETE this test, do not
//      "fix" it to match the new behavior. Its entire premise expires with the
//      placeholder it documents.
//
// COLLATERAL: if _updateLastSeen() ever does get gated, three other tests
// dereference `_lastSeenOpponentPos` with no null guard of their own and will
// also go red with a bare, unexplained TypeError — the "_updateLastSeen
// stores a Vec3-style clone..." test just above this block, "TC13: in
// exhibition mode ATTACK, the last-seen memory and the observation all
// follow the human entity (AC2)", and "TC22: a challenger who leaves
// mid-match zeroes the opponent block, keeps the memory, and never throws".
// Only THIS test explains itself; those three will look like unrelated
// flakes unless whoever is gating already knows to expect them.
// ===========================================================================
test('TC14 (AC11): _updateLastSeen still writes memory when the opponent is behind the learner — KNOWN CONTRACT VIOLATION frozen through 2026-08-20 (TODO(T12))', () => {
  const bots = new ArenaBots({}, { transport: { send: () => {} } });

  // The learner stands at the origin facing +z (yaw 0 is the "looks toward
  // +z" convention documented in env/perception_filter.py). _updateLastSeen()
  // never actually reads this learner entity — that is exactly the bug — so
  // it is set up here only to make "behind the learner" a concrete geometry
  // for a human reader, not because the code under test consults it.
  bots.learner = { entity: { position: { x: 0.5, y: 64, z: 0.5 }, yaw: 0, pitch: 0 } };

  // The opponent is 20 blocks directly BEHIND the learner: 180 degrees off
  // its facing, nowhere near env/perception_filter.py's 70-degree FOV cone by
  // any measure. A genuinely visibility-gated implementation would refuse to
  // update memory from this position.
  const behindPos = {
    x: 0.5,
    y: 64,
    z: -19.5,
    clone() {
      return { x: this.x, y: this.y, z: this.z };
    },
  };
  bots.dummy = { entity: { position: behindPos } };

  bots._updateLastSeen();

  // Asserted in two steps on purpose: a visibility gate leaves
  // _lastSeenOpponentPos null, and reading .x/.y/.z off null throws a generic
  // TypeError that would bury the explanatory message below. Check non-null
  // FIRST so a future gater sees this message, not a stack trace that reads
  // like a flaky test.
  assert.ok(
    bots._lastSeenOpponentPos !== null,
    '_updateLastSeen must still record the live position even when the ' +
      'opponent is behind the learner and outside any FOV cone — this is the ' +
      'frozen placeholder (TODO(T12), AC11), not a bug to fix here. If it is ' +
      'after 2026-08-20 and TODO(T12) has been resolved with real ' +
      'PerceptionFilter-backed gating, DELETE this test instead of updating it.',
  );
  assert.deepEqual(
    { x: bots._lastSeenOpponentPos.x, y: bots._lastSeenOpponentPos.y, z: bots._lastSeenOpponentPos.z },
    { x: 0.5, y: 64, z: -19.5 },
    '_updateLastSeen must snapshot the opponent\'s LIVE position, not some other value',
  );
});

test('a close message drops only the client; bots stay in-game (per-episode close)', async () => {
  const drops = [];
  const transport = {
    send: () => {},
    dropConnection: () => drops.push('drop'),
    close: () => {
      throw new Error('full transport close must NOT run on a close message');
    },
  };
  const bots = new ArenaBots({}, { transport });
  let quits = 0;
  bots.learner = { ...mockBot('learner_bot'), quit: () => { quits += 1; } };
  bots.dummy = { ...mockBot('dummy_bot'), quit: () => { quits += 1; } };

  await bots._handleMessage({ type: 'close' });

  assert.deepEqual(drops, ['drop'], 'only the client connection is dropped');
  assert.equal(quits, 0, 'neither bot quits on a per-episode close');
});

/** Gate options driving an instant timeout via an injected clock (no real wait). */
function instantTimeoutGate() {
  let t = 0;
  return {
    timeoutMs: 100,
    pollIntervalMs: 10,
    now: () => {
      const v = t;
      t += 60; // two polls exceed timeoutMs=100 -> immediate timeout
      return v;
    },
    sleep: async () => {},
  };
}

test('handleReset after a FAILED gate sends ok:false and NO state (env retries)', async () => {
  const sent = [];
  // Drive the gate with an injected clock + no-op sleep so the never-matching
  // learner fails the gate INSTANTLY instead of burning the real 3 s timeout.
  const bots = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: instantTimeoutGate(),
  });
  bots.learner = mockBot('learner_bot', { inventory: [] });
  bots.dummy = mockBot('dummy_bot');

  await bots.handleReset({ type: 'reset', episode: 0, seed: 0 });

  // A stray state after ok:false would desync the env's retry exchange.
  assert.equal(sent.length, 1);
  assert.equal(sent[0].type, 'reset_ack');
  assert.equal(sent[0].ok, false);
});

test('handleReset skips the reply (no bridge error) when the client disconnects during the gate', async () => {
  const sent = [];
  const errors = [];
  // isConnected passes the guard, but the write throws like BridgeServer.send on
  // a dead socket (the client disconnected during the gate — a TOCTOU drop).
  const transport = {
    isConnected: true,
    send: () => {
      throw new Error('cannot send: no active bridge connection');
    },
    emit: (event, payload) => {
      if (event === 'error') errors.push(payload);
    },
  };
  const bots = new ArenaBots({}, { transport, readbackOptions: instantTimeoutGate() });
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockBot('dummy_bot');

  // The reply send fails, but a gone client is not a bridge fault: handleReset
  // must resolve, send nothing, and NOT report a bridge 'error'.
  await assert.doesNotReject(() =>
    bots.handleReset({ type: 'reset', episode: 0, seed: 0 }),
  );
  assert.deepEqual(sent, [], 'nothing reaches a disconnected client');
  assert.deepEqual(errors, [], 'a disconnect during the gate is not a bridge error');
});

// ===========================================================================
// state.tick source: the OUTBOUND tick is the learner's SERVER world age
// (bot.time.age), which Mineflayer sets only from the update_time packet, so it
// tracks the real server tick rate rather than the client physicsTick counter.
// It falls back to the internal per-step counter before the first update_time
// packet (and for fake bots with no `time`). The step path is driven via an
// injected _waitTicksImpl, matching the handleStep harness in actions.test.js.
// ===========================================================================

/** A minimal learner/dummy the step path can snapshot (no live server). */
function stepBot(username, { age } = {}) {
  const bot = {
    username,
    health: MAX_HEALTH,
    heldItem: { name: 'iron_sword' },
    entity: {
      username,
      position: { x: 0.5, y: 64, z: 0.5 },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
      pitch: 0,
      onGround: true,
    },
    on() {},
    off() {},
    once() {},
    setControlState() {},
    clearControlStates() {},
    attack() {},
    lookAt() {},
    chat() {},
  };
  // The server-authoritative world age, present only once an update_time packet
  // has arrived. Left off (bot.time undefined) to model the pre-packet window.
  if (age !== undefined) {
    bot.time = { age };
  }
  return bot;
}

test('handleStep stamps state.tick from the learner server world-age when it is set', async () => {
  const sent = [];
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  arena.learner = stepBot('learner_bot', { age: 777 });
  arena.dummy = stepBot('dummy_bot');
  arena._waitTicksImpl = async () => {};

  await arena.handleStep({ type: 'step', action: 0 });

  assert.equal(sent.length, 1);
  assert.equal(sent[0].type, 'state');
  assert.doesNotThrow(() => validateOutbound(sent[0]));
  // The wire tick is the real server world age, NOT the internal per-step counter
  // (which would have advanced to ACTION_REPEAT for this window).
  assert.equal(sent[0].tick, 777);
});

test('handleStep falls back to the internal tick counter when the learner has no world-age yet', async () => {
  const sent = [];
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  // No `time` on the fake learner: models the window before the first
  // update_time packet arrives (and the unit-test fake with no clock).
  arena.learner = stepBot('learner_bot');
  arena.dummy = stepBot('dummy_bot');
  arena._waitTicksImpl = async () => {};

  await arena.handleStep({ type: 'step', action: 0 });

  assert.equal(sent.length, 1);
  assert.doesNotThrow(() => validateOutbound(sent[0]));
  // Falls back to the monotonic per-step counter advanced to the window boundary.
  assert.equal(sent[0].tick, ACTION_REPEAT);
});

test('handleReset stamps the post-reset state.tick from the learner world-age when set', async () => {
  const sent = [];
  const bots = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockDummy();
  // Server world age from an update_time packet; the post-reset first observation
  // must carry it, not the just-reset internal counter (which is 0).
  bots.learner.time = { age: 4242 };

  await bots.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent.length, 2);
  assert.equal(sent[1].type, 'state');
  assert.equal(sent[1].tick, 4242);
});

// ===========================================================================
// THE OPPONENT HANDLE AGAINST A HUMAN (T1, AC2).
//
// ATTACK and TURN_TO_LAST_SEEN must work against a human PLAYER ENTITY, not
// only against the dummy bot — "they must not silently no-op" is the whole of
// AC2. Before the handle seam both read `this.dummy`, which stays null for the
// entire run in 'human' mode, so both would have no-opped through a live
// classroom demo with nothing in the log to say so. Code-tracing and a manual
// rehearsal were the only cover; these tests are the automated half.
//
// The second half of TC13 pins the OTHER failure the seam can produce:
// _opponentHandle() is stateless and re-resolves on every call, so a window
// that resolves it separately at start and end can attack one person and turn
// toward another the moment the entity map changes mid-window.
//
// MOCK FIDELITY: the challenger fake is a player ENTITY and carries NO health.
// Mineflayer never populates it for anyone but the connected bot (see the note
// at the top of this file), and a fake with health here would paper over
// exactly the damage-channel gap the plan accepts for the demo.
// ===========================================================================

/** A Vec3-ish position: mineflayer entity positions carry clone(). */
function livePosition(x, y, z) {
  return {
    x,
    y,
    z,
    clone() {
      return livePosition(this.x, this.y, this.z);
    },
  };
}

/** A challenger as the learner's own entity view carries one (no health). */
function playerEntity(username, x, z) {
  return {
    type: 'player',
    username,
    position: livePosition(x, 64, z),
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    onGround: true,
  };
}

/** The x/y/z of a remembered position, without its Vec3 methods. */
function coordsOf(pos) {
  return { x: pos.x, y: pos.y, z: pos.z };
}

/**
 * An exhibition arena: 'human' opponent mode, a learner that sees `entities`
 * and records every bot.attack target, the REAL MacroExecutor, and the tick
 * wait injected. `dummy` stays null — in 'human' mode there is no second
 * connection to make, which is the condition under test.
 */
function exhibitionArena(sent, entities, config = {}) {
  const arena = new ArenaBots(
    { opponentMode: OPPONENT_MODE_HUMAN, ...config },
    { transport: { send: (msg) => sent.push(msg) } },
  );
  const learner = stepBot('learner_bot', { age: 100 });
  learner.entities = entities;
  // Record what the REAL executor swings at, the way chatLog records commands.
  learner.attacked = [];
  learner.attack = (entity) => learner.attacked.push(entity);
  arena.learner = learner;
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};
  return arena;
}

test('TC13: in exhibition mode ATTACK, the last-seen memory and the observation all follow the human entity (AC2)', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const bystander = playerEntity('classmate_2', -9.5, 0.5);
  const arena = exhibitionArena(sent, { 11: challenger });

  // The slot is EMPTY until a decision window claims it (T3's latch, tested in
  // its own block below); resolution before that is null by design, so that no
  // packet handler can ever be the thing that decides who the opponent is.
  assert.equal(arena.dummy, null, "'human' mode never spawns a second bot");
  assert.equal(arena._opponentHandle(), null, 'nothing is the opponent until it is claimed');
  arena._claimChallenger();

  // The handle IS the player entity, and it reports no health SOURCE — not a
  // health of zero. A human's health is unreadable; T2's scoreboard owns wins.
  const handle = arena._opponentHandle();
  assert.equal(handle.entity, challenger);
  assert.equal(handle.isBot, false);
  assert.equal(handle.username, 'classmate_1', 'the name comes off the entity itself');
  assert.equal(handle.healthSource, OPPONENT_HEALTH_UNAVAILABLE);
  assert.equal(arena._opponentEntity(), challenger);
  assert.equal(arena._opponentHealth(), null, 'null is "no reading", never 0 health');

  await arena.handleStep({ type: 'step', action: Macro.ATTACK });

  // AC2, positively. A silent no-op here — the executor swinging at null — is
  // the regression: the demo agent would flail at a human all match long.
  assert.equal(arena.learner.attacked.length, 1, 'ATTACK must not no-op against a human');
  assert.equal(arena.learner.attacked[0], challenger, 'the swing targets the PLAYER entity');
  assert.equal(arena.executor.lastSwingTick, 0, 'the swing was stamped, not skipped');
  // TURN_TO_LAST_SEEN's memory is written from that same entity.
  assert.deepEqual(coordsOf(arena._lastSeenOpponentPos), { x: 5.5, y: 64, z: 0.5 });
  // ...and so is the observation, with health 0 for want of a source.
  const first = sent[sent.length - 1];
  assert.doesNotThrow(() => validateOutbound(first));
  assert.deepEqual(first.opponent.pos, [5.5, 64, 0.5]);
  assert.equal(first.opponent.health, 0);

  // ONE resolution per decision window. A second player is standing in the pad
  // and the challenger leaves DURING the window: re-resolving at window end
  // would hand back the bystander, and the agent would swing at one person and
  // turn toward another inside a single step.
  arena.learner.entities[22] = bystander;
  arena._waitTicksImpl = async () => {
    delete arena.learner.entities[11];
  };

  await arena.handleStep({ type: 'step', action: Macro.IDLE });

  assert.deepEqual(
    coordsOf(arena._lastSeenOpponentPos),
    { x: 5.5, y: 64, z: 0.5 },
    'the memory stayed with whoever the window began against',
  );
  assert.deepEqual(
    sent[sent.length - 1].opponent.pos,
    [5.5, 64, 0.5],
    'the observation describes the same person the window opened on',
  );
  // REVISED BY T3, deliberately. Until the first-claimant latch landed this
  // line asserted the OPPOSITE — that the next window resolves to the
  // bystander — pinning the pre-latch statelessness with a comment saying T3
  // would change it. It has: the slot belongs to classmate_1 until a reset, so
  // a challenger who walks out does NOT hand the match to whoever is standing
  // nearby. That is the whole of the latch, and since T2 it is also what keeps
  // the bystander's next death from being credited as the agent's win.
  assert.equal(arena._claimedChallenger, 'classmate_1', 'the slot is still claimed');
  assert.equal(
    arena._opponentHandle(),
    null,
    'the claimant is gone, so there is no opponent — the bystander does not inherit one',
  );
});

// ===========================================================================
// THE FIRST-CLAIMANT LATCH (T3, plan Error Handling: "Two people in the pad").
//
// `_resolveChallengerEntity()` used to be stateless: with challengerUsername
// null it returned whichever non-own player `Object.keys(learner.entities)`
// yielded first, which is deterministically the LOWEST ENTITY ID and not "first
// to enter the pad" — re-decided on every call, over a view that reaches far
// past the arena walls.
//
// T2's review escalated what that costs. `rl_deaths` is server-wide, so once
// scoreboard death detection is live a bystander who resolves as the opponent
// and then dies TO ANYTHING — fall, lava, another player, a different pad's
// learner — is credited as this agent's win. The failure moved from "we aim at
// the wrong person" (on screen, recoverable) to "we announce a win we did not
// earn" (silent, and squarely against AC3).
//
// Two independent properties, tested independently below so one fixture cannot
// stand in for both:
//   (a) THE LATCH — the first claimant holds the slot until a reset;
//   (b) THE PAD GATE — only somebody inside the pad may claim it at all.
// ===========================================================================

test('isInsidePad accepts the pad interior arena:setup_pad actually builds, and nothing else', () => {
  // Bounds mirror setup_pad.mcfunction's own "EXACT BOUNDS" header: standable
  // blocks x in [A.x-7, A.x+15], z in [A.z-11, A.z+11], inside an 8-block air
  // column at y=64..71. The anchor is deliberately NOT (0,0) here — an
  // implementation that ignored the anchor would pass at the origin.
  const anchor = { x: 512, z: 1024 };
  const at = (dx, y, dz) => isInsidePad({ x: anchor.x + dx, y, z: anchor.z + dz }, anchor);

  assert.equal(at(0, 64, 0), true, 'the learner spawn cell is in the pad');
  assert.equal(at(3, 64, 0), true, "the dummy's cell is in the pad");
  assert.equal(at(-7, 64, -11), true, 'the far corner of the walkable floor');
  assert.equal(at(15.7, 65.25, 11.7), true, 'a jumping player hugging the far wall');

  // Outside the bedrock ring on each axis.
  assert.equal(at(-7.5, 64, 0), false, 'past the west wall');
  assert.equal(at(16.5, 64, 0), false, 'past the east wall');
  assert.equal(at(0, 64, -11.5), false, 'past the north wall');
  assert.equal(at(0, 64, 12.5), false, 'past the south wall');
  // THE Y BAND IS NOT DECORATION. The pads float at y=62..71 while the superflat
  // world's own ground is at y=-61..-64, so an x/z-only test would claim
  // somebody standing directly UNDER the arena, hundreds of blocks below it.
  assert.equal(at(0, -60, 0), false, 'the flat world floor beneath the pad is not the pad');
  assert.equal(at(0, 200, 0), false, 'nor is the sky above it');

  // An unreadable position must refuse the claim, never throw inside a decision
  // window and never claim by accident.
  for (const bad of [null, undefined, {}, { x: 0, y: 64 }, { x: NaN, y: 64, z: 0 }]) {
    assert.equal(isInsidePad(bad, anchor), false);
  }
  assert.equal(isInsidePad({ x: 0, y: 64, z: 0 }, null), false);
  // The constants are exported so a launcher/doc cannot restate them and drift.
  assert.equal(PAD_INTERIOR_BOUNDS.minDx, -7);
  assert.equal(PAD_INTERIOR_BOUNDS.maxDz, 12);
});

test('the FIRST claimant holds the slot: a later joiner IN THE PAD is ignored until reset', async () => {
  const sent = [];
  const first = playerEntity('classmate_1', 5.5, 0.5);
  // Entity id 22 > 11, so `first` wins on entity order too — the ids are then
  // SWAPPED below, which is what proves the slot is held by the claim and not
  // re-decided by `Object.keys` order on every call.
  const second = playerEntity('classmate_2', 8.5, 4.5);
  const arena = exhibitionArena(sent, { 11: first });

  assert.equal(arena._opponentHandle(), null, 'nothing is claimed before the first window');

  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(arena._claimedChallenger, 'classmate_1');

  // A second person walks INTO THE PAD mid-match — the one-challenger-at-a-time
  // case. They are eligible in every respect except that the slot is taken.
  arena.learner.entities[2] = second;
  await arena.handleStep({ type: 'step', action: Macro.IDLE });

  assert.equal(arena._claimedChallenger, 'classmate_1', 'the claim does not move');
  assert.equal(arena._opponentHandle().entity, first, 'nor does the opponent');
  assert.deepEqual(sent[sent.length - 1].opponent.pos, [5.5, 64, 0.5]);

  // ...and it is the RESET that arms the next challenger, per the protocol.
  arena._readbackOptions = SINGLE_POLL_GATE;
  await arena.handleReset({ type: 'reset', episode: 1, seed: 0 });
  assert.equal(arena._claimedChallenger, null, 'the reset released the slot');

  delete arena.learner.entities[11];
  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(arena._claimedChallenger, 'classmate_2', 'the next match claims afresh');
});

test('a player OUTSIDE the pad never claims the slot, however alone they are', async () => {
  const sent = [];
  // In the learner's entity view — mineflayer tracks players well past the pad
  // walls — but standing outside the arena. Nobody else is present at all, so
  // a stateless resolver hands them the whole match.
  const distant = playerEntity('classmate_9', -9.5, 0.5);
  const arena = exhibitionArena(sent, { 11: distant });

  await arena.handleStep({ type: 'step', action: Macro.IDLE });

  assert.equal(arena._claimedChallenger, null, 'out of the pad, out of the match');
  assert.equal(arena._opponentHandle(), null);
  assert.deepEqual(sent[sent.length - 1].opponent.pos, [0, 0, 0], 'a zeroed opponent block');

  // They walk in. `livePosition` is the live vector mineflayer mutates, so this
  // is how a real approach looks.
  distant.position.x = 5.5;
  await arena.handleStep({ type: 'step', action: Macro.IDLE });

  assert.equal(arena._claimedChallenger, 'classmate_9', 'entering the pad claims it');
  assert.equal(arena._opponentHandle().entity, distant);
});

test('AC3: an unclaimed bystander who dies outside the pad is NOT the agent\'s win', async () => {
  const sent = [];
  // THE FAILURE THIS TASK EXISTS TO CLOSE. Nobody has claimed the slot, and the
  // only player in the learner's entity view is somebody outside the arena. Two
  // separate mistakes each hand them the match: resolving the "first" player
  // whenever the slot is empty, or claiming without the pad test. Either way
  // their next death — to a mob, a fall, another player, anything — arrives on
  // the server-wide `rl_deaths` objective and is announced as the agent's kill.
  const bystander = playerEntity('classmate_9', -9.5, 0.5);
  const arena = deathArena(sent, { 11: bystander });
  await armDeathDetection(arena);

  await stepOnce(arena, sent);
  assert.equal(arena._claimedChallenger, null, 'nobody in the pad, nobody claimed');

  emitDeathScore(arena, 'classmate_9', 1);

  assert.equal(
    (await stepOnce(arena, sent)).events.opponent_died,
    false,
    'a death outside the match must never be reported as a win',
  );

  // The same person, once they are actually IN the pad and claimed, is the
  // opponent and their death does count — the gate refuses wins, it does not
  // break them.
  bystander.position.x = 5.5;
  await stepOnce(arena, sent);
  assert.equal(arena._claimedChallenger, 'classmate_9');
  emitDeathScore(arena, 'classmate_9', 2);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('the claim survives a disconnect: the same person resumes, nobody else inherits', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const opportunist = playerEntity('classmate_2', 8.5, 4.5);
  const arena = exhibitionArena(sent, { 11: challenger });

  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(arena._claimedChallenger, 'classmate_1');

  // They rage-quit mid-match and somebody else steps into the pad. The plan is
  // explicit that a leaver is NOT a death (entity-gone cannot tell the two
  // apart), so the match is simply held — it is not handed to the newcomer.
  delete arena.learner.entities[11];
  arena.learner.entities[2] = opportunist;
  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(arena._opponentHandle(), null, 'held, not reassigned');
  assert.equal(arena._claimedChallenger, 'classmate_1');

  // They reconnect: mineflayer hands the learner a NEW entity for them (a new
  // id, a new object). Resolution is by NAME, so the match simply resumes.
  arena.learner.entities[77] = playerEntity('classmate_1', 6.5, 1.5);
  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(arena._opponentHandle().entity, arena.learner.entities[77]);
  assert.deepEqual(sent[sent.length - 1].opponent.pos, [6.5, 64, 1.5]);
});

test('a pinned challengerUsername IS the claim: it needs no pad entry and no window', () => {
  const sent = [];
  // Pinned, and NOBODY is in the entity view. The pin is the operator naming
  // the opponent up front, which outranks any latch — and it must keep working
  // with an empty pad, because T2 attributes a death by name alone.
  const arena = exhibitionArena(sent, {}, { challengerUsername: 'classmate_1' });

  assert.equal(arena._challengerSlot(), 'classmate_1', 'the pin fills the slot immediately');
  arena._claimChallenger();
  assert.equal(arena._claimedChallenger, null, 'a pin claims nothing of its own');

  // A pin is not a licence to resolve the wrong entity, though: only the pinned
  // name resolves, wherever anyone else is standing.
  arena.learner.entities[11] = playerEntity('classmate_2', 5.5, 0.5);
  assert.equal(arena._opponentHandle(), null);
  arena.learner.entities[22] = playerEntity('classmate_1', 6.5, 0.5);
  assert.equal(arena._opponentHandle().username, 'classmate_1');
});

test('an operator typo pinning one of our OWN bots resolves to nothing, never to the agent', () => {
  const sent = [];
  // `--challenger-username learner_bot`. run.js cannot catch this (it is a
  // perfectly well-formed username), and the cost of resolving it would be the
  // agent taking ITSELF as the opponent: ATTACK aimed at its own entity and,
  // via `rl_deaths`, its own death credited to it as a win.
  const arena = exhibitionArena(sent, {}, { challengerUsername: 'learner_bot' });
  arena.learner.entities[11] = playerEntity('learner_bot', 5.5, 0.5);

  assert.equal(arena._opponentHandle(), null);
  assert.equal(arena._isChallengerName('learner_bot'), false);
});

test('with no challenger the agent HOLDS IDLE — it does not jog around an empty pad', async () => {
  const sent = [];
  const arena = exhibitionArena(sent, {});
  const pressed = [];
  arena.learner.setControlState = (state, value) => pressed.push([state, value]);

  // A null handle already makes ATTACK a no-op (nothing to swing at), but the
  // movement macros need no opponent at all — so between challengers a live
  // policy would walk the agent around an empty arena on a projector.
  await arena.handleStep({ type: 'step', action: Macro.APPROACH });

  assert.deepEqual(
    pressed.filter(([, value]) => value === true),
    [],
    'no control state may be PRESSED with nobody to fight',
  );
  const state = sent[sent.length - 1];
  assert.doesNotThrow(() => validateOutbound(state));
  assert.deepEqual(state.opponent.pos, [0, 0, 0], 'and the opponent block stays zeroed');

  // The override is exactly as wide as the reason for it: once somebody claims
  // the slot, the requested macro runs untouched.
  arena.learner.entities[11] = playerEntity('classmate_1', 5.5, 0.5);
  await arena.handleStep({ type: 'step', action: Macro.APPROACH });
  assert.ok(
    pressed.some(([state_, value]) => state_ === 'forward' && value === true),
    'APPROACH must run normally against a real challenger',
  );
});

test('BOT mode never claims, never holds IDLE, and never pays for the latch', async () => {
  const sent = [];
  // The M2/training path. Every line T3 added is gated on the opponent MODE, so
  // a missing dummy here must behave exactly as it did before this task: the
  // requested macro runs, and no challenger machinery engages at all.
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  arena.learner = stepBot('learner_bot', { age: 10 });
  arena.learner.entities = { 11: playerEntity('classmate_1', 5.5, 0.5) };
  const pressed = [];
  arena.learner.setControlState = (state, value) => pressed.push([state, value]);
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};

  await arena.handleStep({ type: 'step', action: Macro.APPROACH });

  assert.equal(arena._claimedChallenger, null, 'a human in view is not an opponent here');
  assert.equal(arena._challengerSlot(), null);
  assert.ok(
    pressed.some(([state_, value]) => state_ === 'forward' && value === true),
    'the training path executes the macro it was sent, dummy or no dummy',
  );
});

test('TC22: a challenger who leaves mid-match zeroes the opponent block, keeps the memory, and never throws', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = exhibitionArena(sent, { 11: challenger });

  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.deepEqual(coordsOf(arena._lastSeenOpponentPos), { x: 5.5, y: 64, z: 0.5 });

  // They quit mid-match: mineflayer drops them from the learner's entity view.
  delete arena.learner.entities[11];

  assert.equal(arena._opponentHandle(), null, 'no challenger, no handle');
  assert.equal(arena._opponentEntity(), null);
  assert.equal(arena._opponentHealth(), null);

  await assert.doesNotReject(() => arena.handleStep({ type: 'step', action: Macro.ATTACK }));

  const state = sent[sent.length - 1];
  assert.doesNotThrow(() => validateOutbound(state));
  // State keeps flowing with a ZEROED opponent block — the wire has no slot for
  // a "waiting for challenger" status, so that goes to the bridge log only.
  assert.deepEqual(state.opponent, {
    pos: [0, 0, 0],
    yaw: 0,
    pitch: 0,
    velocity: [0, 0, 0],
    health: 0,
  });
  assert.equal(arena.learner.attacked.length, 0, 'nothing to swing at');
  assert.equal(arena.executor.lastSwingTick, null, 'a swing at nobody never starts the cooldown');
  // The memory is RETAINED, not cleared: TURN_TO_LAST_SEEN must still be able
  // to face where the challenger was last seen (only handleReset clears it).
  assert.deepEqual(coordsOf(arena._lastSeenOpponentPos), { x: 5.5, y: 64, z: 0.5 });
});

// ===========================================================================
// HUMAN DEATH DETECTION VIA THE `rl_deaths` SCOREBOARD (T2, TC15, AC3).
//
// A human challenger has no Mineflayer connection, so neither of the two
// signals that make opponent_died work in 'bot' mode exists for them: no
// `death` event, and no health channel at all (mineflayer never populates
// entity.health for non-self players — the constraint PR #32 worked around by
// reading the dummy's OWN connection). The server-side `deathCount` scoreboard
// is the only remaining source, and these tests are what keep it honest.
//
// MOCK FIDELITY. The packet fakes below use the 1.21.1 field names as
// minecraft-data defines them (`scoreboard_score` = {itemName, scoreName,
// value}; `reset_score` = {entity_name, objective_name}), NOT invented ones,
// and they are emitted on `learner._client` because mineflayer's own
// `scoreUpdated` event is provably dead on this version (its plugin gates on a
// `packet.action` field 1.20.3 deleted). A fake that emitted the plugin event
// would be testing a channel the server can never drive.
// ===========================================================================

/**
 * An exhibition arena whose learner carries a raw packet client, wired exactly
 * as connect() wires it. `dummy` stays null throughout — "no opponent bot
 * connection" is the condition AC3 is about.
 */
function deathArena(sent, entities, config = {}) {
  const arena = new ArenaBots(
    { opponentMode: OPPONENT_MODE_HUMAN, ...config },
    // A short read-back budget: the unconfirmed path is a real case worth
    // testing and must not cost 5 s of wall clock to reach.
    { transport: { send: (msg) => sent.push(msg) }, deathObjectiveTimeoutMs: 20 },
  );
  const learner = stepBot('learner_bot', { age: 100 });
  learner.entities = entities;
  learner.attacked = [];
  learner.attack = (entity) => learner.attacked.push(entity);
  learner.chatLog = [];
  learner.chat = (cmd) => learner.chatLog.push(cmd);
  learner._client = new EventEmitter();
  arena.learner = learner;
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};
  arena.wireDamageEvents();
  return arena;
}

/** The server pushing one holder's new `rl_deaths` value. */
function emitDeathScore(arena, itemName, value, scoreName = RL_DEATHS_OBJECTIVE) {
  arena.learner._client.emit('scoreboard_score', { itemName, scoreName, value });
}

/** The server announcing the objective (action 0 = created/now tracked). */
function emitObjective(arena, action = 0, name = RL_DEATHS_OBJECTIVE) {
  arena.learner._client.emit('scoreboard_objective', { name, action });
}

/**
 * Arm detection the way connect() does: the objective is echoed back, so
 * everything after this point is live news rather than a replay of history.
 *
 * The echo is answered FROM the `setdisplay` command and on a timer, not
 * emitted before the call. _verifyDeathObjective re-arms its own latch on entry
 * (S1) precisely so a second invocation cannot confirm itself from the previous
 * run's echo — a fixture that pre-set the latch would be exercising exactly the
 * stale-latch path that fix removes.
 */
async function armDeathDetection(arena) {
  const inner = arena.learner.chat;
  arena.learner.chat = (cmd) => {
    inner.call(arena.learner, cmd);
    if (typeof cmd === 'string' && cmd.startsWith('/scoreboard objectives setdisplay')) {
      setTimeout(() => emitObjective(arena), 1);
    }
  };
  let confirmed;
  try {
    confirmed = await arena._verifyDeathObjective();
  } finally {
    arena.learner.chat = inner;
  }
  assert.equal(confirmed, true, 'the fixture must leave detection CONFIRMED');
  return confirmed;
}

/** Run one decision window and hand back the `state` it emitted. */
async function stepOnce(arena, sent, action = Macro.IDLE) {
  await arena.handleStep({ type: 'step', action });
  const state = sent[sent.length - 1];
  assert.doesNotThrow(() => validateOutbound(state));
  return state;
}

test('the death objective is added AND pinned to a display slot (no slot, no packets)', () => {
  // THE SECOND COMMAND IS NOT COSMETIC. Decompiling the pinned jar
  // (`javap -p -c` on server/versions/1.21.1/paper-1.21.1.jar) shows
  // ServerScoreboard.onScoreChanged broadcasting ClientboundSetScorePacket only
  // inside `if (trackedObjectives.contains(objective))`, and the sole caller of
  // startTrackingObjective in the whole jar is setDisplayObjective. An objective
  // that is merely ADDED therefore emits no packet to anyone: the server counts
  // the deaths and the bridge is told nothing, with no error to notice. Deleting
  // the setdisplay line must fail here rather than during a live exhibition.
  assert.deepEqual(formatDeathObjectiveCommands(), [
    '/scoreboard objectives add rl_deaths deathCount',
    '/scoreboard objectives setdisplay list rl_deaths',
  ]);
  // The objective NAME is a cross-process contract (the runbook and any operator
  // command say `rl_deaths`), so it is pinned as a literal, not via the export.
  assert.equal(RL_DEATHS_OBJECTIVE, 'rl_deaths');
});

test('TC15: a human challenger\'s FIRST death reaches the wire as opponent_died (AC3)', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });
  await armDeathDetection(arena);

  assert.equal(arena.dummy, null, 'AC3 is about winning with NO opponent bot connection');
  // The window that establishes who the opponent is.
  const before = await stepOnce(arena, sent);
  assert.equal(before.events.opponent_died, false, 'nobody has died yet');

  // A deathCount entry does not exist until the first death, so the very first
  // packet a challenger ever produces is `value: 1` — AT the moment they die.
  // Treating a first sighting as a baseline would swallow exactly this event.
  emitDeathScore(arena, 'classmate_1', 1);

  const state = await stepOnce(arena, sent);
  assert.equal(state.events.opponent_died, true, 'the death must reach the wire');
  // ...and exactly once: it belongs to the window it happened in.
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, false);
});

test('the death baseline SURVIVES a reset — a re-sent score is not a second win', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });

  // Boot-time replay: `setdisplay` makes the server resend every pre-existing
  // score, so a challenger who died in an EARLIER match arrives at 3. Detection
  // is not armed yet, so this is a baseline and not three wins.
  emitDeathScore(arena, 'classmate_1', 3);
  assert.equal(arena.events.peek().opponent_died, false, 'a replay is history, not news');
  await armDeathDetection(arena);

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  // A straggler re-send of the SAME value after the reset. This is the case
  // that fails if anyone adds `this._deathScores.clear()` to handleReset: with
  // the baseline dropped, an unchanged 3 reads as three-deaths-from-zero and
  // the agent is handed a win it never earned, before the match even starts.
  emitDeathScore(arena, 'classmate_1', 3);
  assert.equal(
    (await stepOnce(arena, sent)).events.opponent_died,
    false,
    'an unchanged score is never a death — the baseline must outlive the reset',
  );

  // The genuine next death still lands.
  emitDeathScore(arena, 'classmate_1', 4);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('only the CHALLENGER\'s deaths count — the learner\'s and a bystander\'s do not', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });
  await armDeathDetection(arena);

  // FIRST, before any step has run: no attribution memory yet, so this is the
  // live-resolve branch — the one a death landing between the reset ack and the
  // episode's first window takes.
  emitDeathScore(arena, 'learner_bot', 1);
  emitDeathScore(arena, 'classmate_2', 1);
  assert.equal(
    arena.events.peek().opponent_died,
    false,
    'with no memory yet, only a live-resolved challenger may be credited',
  );

  await stepOnce(arena, sent);

  // `rl_deaths` is server-wide: the learner's own deaths land on it too, and so
  // does every bystander's and every other pad's bots'. Crediting any of them
  // would hand the agent a win for someone else's mistake — and the learner's
  // own entry would credit it for DYING.
  emitDeathScore(arena, 'learner_bot', 2);
  emitDeathScore(arena, 'dummy_bot', 1);
  emitDeathScore(arena, 'classmate_2', 2);
  emitDeathScore(arena, 'classmate_1', 1, 'some_other_objective');

  assert.equal(
    (await stepOnce(arena, sent)).events.opponent_died,
    false,
    'no death may be attributed to anyone but the opponent',
  );

  emitDeathScore(arena, 'classmate_1', 1);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('a scoreboard death is NOT gated by _suppressOpponentEvents', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });
  await armDeathDetection(arena);
  await stepOnce(arena, sent);

  // The same deliberate asymmetry the dummy's `death` handler has (see the
  // comment in wireDamageEvents): the flag gates HEALTH events, because a reset
  // heals asynchronously and those deltas are not combat. A death is not a
  // delta. Reset-window deaths are discarded by the winning handleReset's
  // events.reset() instead — gating here would silently drop a real win
  // whenever a retry reset happened to be in flight.
  arena._suppressOpponentEvents = true;
  emitDeathScore(arena, 'classmate_1', 1);

  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('a death is still attributed when the challenger\'s entity vanishes with them', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const entities = { 11: challenger };
  const arena = deathArena(sent, entities);
  await armDeathDetection(arena);

  // One window fixes who the opponent is.
  await stepOnce(arena, sent);
  // Dying can drop them out of the learner's entity view in the same instant,
  // and _opponentHandle() is stateless — so a packet-time live resolve alone
  // would lose the win in the exact moment it was earned.
  delete entities[11];
  assert.equal(arena._opponentHandle(), null, 'the entity really is gone');

  emitDeathScore(arena, 'classmate_1', 1);

  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('a pinned challengerUsername is the ONLY name that can win the match', async () => {
  const sent = [];
  const arena = deathArena(sent, {}, { challengerUsername: 'classmate_1' });
  await armDeathDetection(arena);

  // Nobody is in the entity view at all, so there is no live resolve and no
  // attribution memory. A pinned name still decides both cases on its own.
  emitDeathScore(arena, 'classmate_2', 1);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, false);

  emitDeathScore(arena, 'classmate_1', 1);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('a removed score entry re-baselines to 0 rather than reporting anything', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });
  await armDeathDetection(arena);
  await stepOnce(arena, sent);

  emitDeathScore(arena, 'classmate_1', 4);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);

  // `/scoreboard players reset` — 1.20.3 split this out of the score packet and
  // mineflayer's plugin handles neither half. An absent entry reads as 0, so the
  // NEXT death is a 0 -> 1 step and must still be reported.
  arena.learner._client.emit('reset_score', {
    entity_name: 'classmate_1',
    objective_name: RL_DEATHS_OBJECTIVE,
  });
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, false, 'a removal is not a death');

  emitDeathScore(arena, 'classmate_1', 1);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

test('the objective read-back is VERIFIED, and an unverified one is named loudly', async () => {
  const sent = [];
  const arena = deathArena(sent, {});
  // Answer the way the server does — asynchronously, after the command has
  // gone out. A fixture that confirmed synchronously would be more capable than
  // the server and would hide a read-back that is never actually awaited.
  arena.learner.chat = (cmd) => {
    arena.learner.chatLog.push(cmd);
    if (cmd.startsWith('/scoreboard objectives setdisplay')) {
      setTimeout(() => emitObjective(arena), 1);
    }
  };

  assert.equal(await arena._verifyDeathObjective(), true);
  assert.deepEqual(arena.learner.chatLog, formatDeathObjectiveCommands());
  assert.equal(arena._deathObjectiveReady, true);

  // A server that never echoes the objective back: the commands may have been
  // rejected outright (this project's failures are silent), so say so — but ARM
  // detection anyway. A disarmed bridge would turn a loud problem into the
  // silent one this whole path exists to avoid.
  const quiet = new ArenaBots(
    { opponentMode: OPPONENT_MODE_HUMAN },
    { transport: { send: () => {} }, deathObjectiveTimeoutMs: 5 },
  );
  quiet.learner = stepBot('learner_bot');
  const logged = [];
  const realError = console.error;
  console.error = (line) => logged.push(line);
  try {
    assert.equal(await quiet._verifyDeathObjective(), false);
  } finally {
    console.error = realError;
  }
  assert.equal(quiet._deathObjectiveReady, true, 'unverified still means armed');
  assert.equal(logged.length, 1);
  assert.match(logged[0], /rl_deaths objective NOT confirmed/);
});

test('the read-back latch is RE-ARMED per call: a second verify cannot confirm itself (S1)', async () => {
  const sent = [];
  const arena = deathArena(sent, {});
  await armDeathDetection(arena);
  assert.equal(arena._deathObjectiveSeen, true, 'the first call really did confirm');

  // A second invocation — a reconnect, or T3's launcher connecting twice —
  // against a server that answers nothing. Without the re-arm it would read the
  // FIRST call's echo, return confirmed on its first poll, and have verified
  // nothing about the commands it just issued: a read-back that has silently
  // stopped reading back, reported as healthy.
  const logged = [];
  const realError = console.error;
  console.error = (line) => logged.push(line);
  try {
    assert.equal(await arena._verifyDeathObjective(), false, 'a silent server is NOT a confirmation');
  } finally {
    console.error = realError;
  }
  assert.match(logged[0], /rl_deaths objective NOT confirmed/);
  // ...and it is still armed, so a real death after a failed re-verify counts.
  assert.equal(arena._deathObjectiveReady, true);
});

/**
 * How late the replayed half of the burst arrives. Must sit strictly between a
 * degenerate drain (a single event-loop yield) and the real one
 * (DEFAULT_READBACK.pollIntervalMs, 50 ms), so the test fails if the drain is
 * removed OR shortened to a token yield, and passes with 40 ms of slack when it
 * is intact.
 */
const REPLAY_DELAY_MS = 10;

test('a score replayed just AFTER the confirmation packet is baseline, not a win (W2)', async () => {
  const sent = [];
  // Pinned, so attribution cannot be what saves the assertion: this name is the
  // one name allowed to win the match.
  const arena = deathArena(sent, {}, { challengerUsername: 'classmate_1' });

  // THE BURST, SPLIT. getStartTrackingPackets builds one list — the objective,
  // its display slot, then one score per existing entry — and the server writes
  // it back to back. Delivered in a single TCP read the poll can only ever see
  // all of it; split across two reads the poll can land in the gap. That gap is
  // what this models: the confirmation is observable at the poll, the scores
  // are not there yet.
  //
  // FIDELITY NOTE — the round trip is deliberately compressed to zero here.
  // Answering `setdisplay` synchronously is not how a server behaves (the test
  // above owns that: it answers on a timer and would catch a read-back that is
  // never awaited). Compressing it is what makes the ORDER under test a fact
  // rather than a race between two wall-clock timers: the confirmation is seen
  // on the poll's first check, and the replay lands strictly afterwards.
  arena.learner.chat = (cmd) => {
    arena.learner.chatLog.push(cmd);
    if (!cmd.startsWith('/scoreboard objectives setdisplay')) {
      return;
    }
    emitObjective(arena);
    // classmate_1 died three times in a match that ended before this process
    // existed. This is history arriving late, not news.
    //
    // 10 ms, not 0: the gap between two TCP reads is I/O time, not a macrotask
    // hop, and a drain that only yielded the event loop once would still arm
    // ahead of this. It must be comfortably inside the 50 ms drain and
    // comfortably outside any degenerate one.
    setTimeout(() => emitDeathScore(arena, 'classmate_1', 3), REPLAY_DELAY_MS);
  };

  assert.equal(await arena._verifyDeathObjective(), true);
  // Let the replay land before asking whether it was believed. Without the
  // drain the flag is already armed by the time it arrives, so an earlier
  // assertion would report "no death" for the wrong reason.
  await new Promise((resolve) => setTimeout(resolve, REPLAY_DELAY_MS * 2 + 5));

  assert.equal(
    arena.events.peek().opponent_died,
    false,
    'a replayed historical score is not a kill — arming must wait out the burst',
  );
  assert.equal(arena._deathScores.get('classmate_1'), 3, 'it still baselines the map');

  // ...and the genuine next death, after the burst, still lands.
  emitDeathScore(arena, 'classmate_1', 4);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

// ===========================================================================
// connect() — THE ONE LINE THAT ARMS HUMAN WIN DETECTION (T2, AC3).
//
// Every test above drives _verifyDeathObjective() by hand, so every one of them
// stays green with the LIFECYCLE call site gone. Replacing connect()'s
// `if (!this._opponentIsBot())` with `if (false)` switches the whole feature
// off — the objective is never added, never tracked, and (per the decompile
// quoted above) no ClientboundSetScorePacket is ever sent to anyone — and the
// suite still reported 157 passing. Nothing bound the arming edge.
//
// It matters because connect() is refactored next, and the symptom of losing
// this line is not an error: it is an exhibition in which nobody ever dies.
//
// MOCK FIDELITY. createBot comes through the injection seam ArenaBots already
// has (deps.createBot), so this exercises the real connect(), not a re-creation
// of it. The fakes are EventEmitters because waitForSpawn attaches
// .once('spawn'/'error'/'kicked'), and they fire `spawn` on a TIMER: createBot
// returns synchronously and connect() attaches that listener afterwards, so a
// fake that spawned synchronously would be missed and connect() would hang.
// The scoreboard echo rides `_client`, the raw packet feed, for the same reason
// the handlers do — mineflayer's own scoreUpdated is dead on 1.21.1.
// ===========================================================================

/** A mineflayer-shaped bot carrying only what connect() actually touches. */
function connectBot(username) {
  const bot = Object.assign(new EventEmitter(), {
    username,
    chatLog: [],
    health: MAX_HEALTH,
    // The raw packet client the scoreboard listeners attach to.
    _client: new EventEmitter(),
  });
  bot.chat = (cmd) => {
    bot.chatLog.push(cmd);
    // The server's answer to `setdisplay` is a PACKET, not a chat reply, and it
    // is a round trip — hence the timer. 0 ms still lands strictly before the
    // read-back's first poll, so the confirmation is deterministic.
    if (typeof cmd === 'string' && cmd.startsWith('/scoreboard objectives setdisplay')) {
      setTimeout(
        () => bot._client.emit('scoreboard_objective', { name: RL_DEATHS_OBJECTIVE, action: 0 }),
        0,
      );
    }
  };
  // Spawn on the next macrotask — see the MOCK FIDELITY note above.
  setTimeout(() => bot.emit('spawn'), 0);
  return bot;
}

/**
 * An ArenaBots whose connect() builds `connectBot`s, plus the usernames it
 * asked for (one bot in 'human' mode, two in 'bot' mode).
 */
function connectFixture(config = {}) {
  const created = [];
  const arena = new ArenaBots(config, {
    createBot: (opts) => {
      const bot = connectBot(opts.username);
      created.push(opts.username);
      return bot;
    },
    transport: { send: () => {} },
    deathObjectiveTimeoutMs: 50,
  });
  return { arena, created };
}

test('connect() in HUMAN mode issues the setup pad command AND both scoreboard commands', async () => {
  const { arena, created } = connectFixture({ opponentMode: OPPONENT_MODE_HUMAN });

  await arena.connect();

  assert.deepEqual(created, ['learner_bot'], "'human' mode never makes a second connection");
  assert.equal(arena.dummy, null);
  // EXACT, not a superset: connect() chats these three commands and nothing
  // else, in this order. BOTH scoreboard commands must be here — `add` alone
  // makes the server count deaths and tell the bridge nothing.
  assert.deepEqual(arena.learner.chatLog, [
    formatSetupPadCommand(arena.padOrigin),
    ...formatDeathObjectiveCommands(),
  ]);
  // The read-back was AWAITED, not left floating: an unawaited promise is
  // process-fatal here, and connect() returning before it resolves would leave
  // the burst racing the first episode.
  assert.equal(arena._deathObjectiveSeen, true, 'the server echo arrived within connect()');
  assert.equal(arena._deathObjectiveReady, true, 'detection is armed by the time connect() returns');
  assert.equal(arena.learner._client.listenerCount('scoreboard_score'), 1);
});

test('connect() in BOT mode issues NEITHER scoreboard command and stays byte-inert', async () => {
  const { arena, created } = connectFixture();

  await arena.connect();

  assert.deepEqual(created, ['learner_bot', 'dummy_bot'], 'the training path still spawns both');
  assert.notEqual(arena.dummy, null);
  // The M2/training path must be byte-identical to before T2 existed: one
  // command, no objective, and not one scoreboard listener.
  assert.deepEqual(arena.learner.chatLog, [formatSetupPadCommand(arena.padOrigin)]);
  assert.deepEqual(arena.dummy.chatLog, []);
  assert.equal(arena._deathObjectiveReady, false, 'nothing is armed on the training path');
  assert.equal(arena.learner._client.listenerCount('scoreboard_score'), 0);
  assert.equal(arena.learner._client.listenerCount('scoreboard_objective'), 0);
});

test('in BOT mode the scoreboard path is entirely absent and the dummy owns opponent_died', async () => {
  const sent = [];
  // The default opponent mode: the M2/training path, which must be byte-identical.
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  arena.learner = liveBot('learner_bot');
  arena.learner._client = new EventEmitter();
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();

  // Not one listener, so a stray packet cannot reach the aggregator even in
  // principle, and no scoreboard command is ever chatted on the training path.
  assert.equal(arena.learner._client.listenerCount('scoreboard_score'), 0);
  assert.equal(arena.learner._client.listenerCount('scoreboard_objective'), 0);
  arena.learner._client.emit('scoreboard_score', {
    itemName: 'dummy_bot',
    scoreName: RL_DEATHS_OBJECTIVE,
    value: 9,
  });
  assert.equal(arena.events.peek().opponent_died, false);
  assert.deepEqual(arena.learner.chatLog, [], 'no /scoreboard command on the training path');

  // opponent_died still comes from where it always has: the dummy's OWN death
  // event on its OWN connection.
  arena.dummy.emit('death');
  assert.equal(arena.events.peek().opponent_died, true);
});

test('re-wiring does not double-count, and a malformed packet never throws', async () => {
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = deathArena(sent, { 11: challenger });
  // A reconnect re-wires; handlers must be removed from the client they were
  // added to first (W1a), or one death would be recorded twice.
  arena.wireDamageEvents();
  arena.wireDamageEvents();
  assert.equal(arena.learner._client.listenerCount('scoreboard_score'), 1);
  await armDeathDetection(arena);
  await stepOnce(arena, sent);

  // Garbage off the wire records nothing and takes nothing down.
  for (const bad of [null, undefined, {}, { scoreName: RL_DEATHS_OBJECTIVE }]) {
    assert.doesNotThrow(() => arena.learner._client.emit('scoreboard_score', bad));
    assert.doesNotThrow(() => arena.learner._client.emit('reset_score', bad));
    assert.doesNotThrow(() => arena.learner._client.emit('scoreboard_objective', bad));
  }
  for (const bad of [NaN, Infinity, '3', null]) {
    emitDeathScore(arena, 'classmate_1', bad);
  }
  assert.equal(
    (await stepOnce(arena, sent)).events.opponent_died,
    false,
    'a non-finite score is not a death',
  );

  emitDeathScore(arena, 'classmate_1', 1);
  assert.equal((await stepOnce(arena, sent)).events.opponent_died, true);
});

// ===========================================================================
// THE DUMMY READ-BACK GATE (plan Error Handling: "reset-generated health
// events").
//
// handleReset heals and teleports the dummy asynchronously. Acking while the
// dummy is still hurt would let the episode's first real hit be measured against
// a phantom baseline — the same shape as the bug this whole plan exists to fix.
// T2's answer is a SECOND gate on the dummy plus seeding _prevOpponentHealth
// from that gate's CONFIRMED read-back rather than from an assumed constant.
//
// Every other reset test here drives the dummy gate only in the PASSING
// direction, where it is satisfied trivially on the first synchronous poll.
// That leaves T2's most safety-critical addition deletable with the suite fully
// green. These two cases drive it in the FAILING direction and pin the seed to
// the read-back, so a future refactor cannot quietly revert either one.
//
// Gate options: `timeoutMs: 0` uses the real clock and runReadbackGate's
// documented "poll at least once even if timeoutMs is 0" path — exactly one
// poll per bot, no wall-clock burned, no fake clock shared between two
// concurrent gates.
// ===========================================================================

const SINGLE_POLL_GATE = Object.freeze({ timeoutMs: 0, pollIntervalMs: 1 });

test('handleReset acks ok:false and sends NO state when only the DUMMY fails its gate', async () => {
  const sent = [];
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: SINGLE_POLL_GATE,
  });
  // The learner is fully reset and WILL confirm.
  arena.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  // The dummy is correctly placed and regeared but the reset's /effect heal has
  // not landed: it is still carrying the previous episode's damage. Health is
  // the only failing dimension, so this isolates the dummy gate's health check.
  arena.dummy = mockDummy();
  arena.dummy.health = 14;

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent.length, 1, 'a hurt dummy blocks the episode: ack only, NO first observation');
  assert.equal(sent[0].type, 'reset_ack');
  assert.equal(
    sent[0].ok,
    false,
    'ok is result.ok AND dummyResult.ok — a confirmed learner alone must not start the episode',
  );
  // The ack's readback stays the LEARNER's (frozen wire shape), and it confirms
  // the learner really did pass — so ok:false can only have come from the dummy.
  assert.equal(sent[0].readback.health, MAX_HEALTH);
  assert.deepEqual(sent[0].readback.inventory, ['iron_sword']);
});

test('handleReset seeds _prevOpponentHealth from the CONFIRMED dummy readback, not an assumed 20', async () => {
  const sent = [];
  // healthEpsilon is widened so the gate CONFIRMS a dummy sitting at 18 rather
  // than 20. That is the whole point: the gate passing does not imply the dummy
  // is at MAX_HEALTH, so the baseline must trace to what the server actually
  // reported. Seeding a constant here would manufacture 2 points of phantom
  // damage on the episode's first hit.
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: { ...SINGLE_POLL_GATE, healthEpsilon: 5 },
  });
  arena.learner = liveBot('learner_bot');
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.dummy.health = 18;
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);
  const spy = spyRecorders(arena);

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent.length, 2, 'both gates confirmed, so the episode starts');
  assert.equal(sent[0].ok, true);
  assert.equal(
    arena._prevOpponentHealth,
    18,
    'the baseline is the confirmed read-back health, not MAX_HEALTH',
  );

  // The behavioral consequence, which is what actually matters: the first hit
  // of the episode takes the dummy 18 -> 12 and must record exactly 6. A
  // baseline seeded to a constant 20 would record 8 — phantom damage from a
  // health delta the learner never dealt.
  arena.dummy.health = 12;
  arena.dummy.emit('health');
  assert.deepEqual(spy.damageDealt, [6], 'exactly one recorder call, for the 6 that really landed');
  assert.equal(arena.events.drain().damage_dealt, 6, 'no phantom damage on the first hit');
});

test('in bot mode a NULL dummy still fails the dummy gate (the gate is keyed on MODE)', async () => {
  // T1 made the dummy gate conditional so it can no-op for a human challenger,
  // who has no connection to read back. Nothing pins the other direction: re-
  // keying that condition from the opponent MODE to `this.dummy !== null` looks
  // equivalent and would silently SKIP the gate whenever the dummy failed to
  // spawn — acking a reset nobody verified, on the training path.
  //
  // Nothing downstream would catch that: _resetWasConfirmed('dummy') RETURNS
  // TRUE when this.dummy is null (its missing-bot branch, deliberate so a human
  // opponent's absent half auto-confirms), so the causality beacon cannot fail
  // the reset either. This gate is the only thing left standing.
  //
  // ok:false alone is therefore not the tell (an unconfirmed reset is also
  // ok:false). WHICH check rejected is: the "NOT confirmed by the datapack"
  // diagnosis is emitted only when both gates MATCHED, so it must stay silent.
  const sent = [];
  const errors = [];
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: SINGLE_POLL_GATE,
  });
  arena.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  // The dummy never connected (a failed spawn, or a reset racing connect()).
  arena.dummy = null;

  const realError = console.error;
  console.error = (msg) => errors.push(String(msg));
  try {
    await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  } finally {
    console.error = realError;
  }

  assert.equal(sent.length, 1, 'ack only: an unverifiable dummy must not start the episode');
  assert.equal(sent[0].type, 'reset_ack');
  assert.equal(sent[0].ok, false);
  assert.ok(
    !errors.some((line) => line.includes('reset NOT confirmed by the datapack')),
    'the DUMMY GATE rejected — a gate that had been skipped would fail later, at the beacon',
  );
});

// ===========================================================================
// handleReset x the damage channel (AC4/AC5).
//
// The reset heals and teleports BOTH bots asynchronously, so the reset window
// generates health and death events that are not combat. Three behaviors are
// pinned here because each guards a silent UNDER-COUNT — a failure mode that is
// invisible from outside the bridge, which is exactly how the damage channel
// stayed dead for the life of the project.
// ===========================================================================

test('a STALE reset that loses the epoch race applies NONE of its post-gate effects', async () => {
  const sent = [];
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  // The learner's /give has not landed yet, so its gate cannot match on the
  // first poll and the handler parks inside runReadbackGate.
  arena.learner = liveBot('learner_bot', { inventory: [] });
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);

  const gate = parkedGate();
  arena._readbackOptions = gate.options;

  // Reset A: in flight, parked mid-gate.
  const resetA = arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  await gate.parked;
  assert.equal(sent.length, 0, 'A has not acked yet');

  // The env's reset path is reconnect-and-retry, so reset B arrives while A is
  // still polling. B's gates match immediately (the gear has since landed) and
  // B owns the episode from here.
  arena.learner.setInventory(['iron_sword']);
  arena._readbackOptions = {};
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  assert.equal(sent.length, 2, 'B acked and sent the post-reset first observation');
  assert.equal(sent[0].ok, true);

  // B's episode lands a real first-window hit.
  arena.dummy.health = MAX_HEALTH - 6;
  arena.dummy.emit('health');
  assert.equal(arena._prevOpponentHealth, 14);

  // Now A finally gets past its gate. It must apply none of its four post-gate
  // effects: no baseline re-seed, no events.reset(), no suppression-flag clear,
  // no ack.
  gate.release();
  await resetA;

  assert.equal(sent.length, 2, 'the stale handler sent no second ack (that would desync the env)');
  assert.equal(arena._suppressOpponentEvents, false, 'the flag is clear, not stranded');
  assert.equal(arena._prevOpponentHealth, 14, 'the stale handler did not re-seed the live baseline');
  assert.equal(
    arena.events.drain().damage_dealt,
    6,
    "the live episode's first-window damage survived the stale handler untouched",
  );
});

test(
  "a dummy 'death' during a reset is discarded by the winning reset's events.reset(), " +
    'NOT by the suppression flag — do not gate the death handler',
  async () => {
    // The suppression flag gates ONLY _onOpponentHealth. Extending it to the
    // death handler would look like a tidy symmetry fix and would silently break
    // mid-episode death detection for every window the flag happens to be up.
    // Both halves are asserted here so that refactor cannot land quietly.
    const sent = [];
    const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
    arena.learner = liveBot('learner_bot', { inventory: [] });
    arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
    arena.wireDamageEvents();
    answerResetLikeTheServer(arena);
    const spy = spyRecorders(arena);

    const gate = parkedGate();
    arena._readbackOptions = gate.options;
    const reset = arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
    await gate.parked;

    // Mid-reset the dummy dies (respawn jank, or the reset's own heal ordering).
    assert.equal(arena._suppressOpponentEvents, true, 'the reset window is suppressed');
    arena.dummy.emit('death');
    assert.equal(
      spy.opponentDied,
      1,
      'the death handler ran despite the flag — it is deliberately NOT gated',
    );

    // The gear lands; the reset completes and discards everything the reset
    // window generated, the death included.
    arena.learner.setInventory(['iron_sword']);
    gate.release();
    await reset;

    assert.equal(sent.length, 2);
    assert.equal(sent[0].type, 'reset_ack');
    assert.equal(sent[0].ok, true);
    assert.equal(
      sent[1].events.opponent_died,
      false,
      'events.reset() discarded the reset-window death before the first observation',
    );

    // And mid-episode death detection is alive immediately afterwards — the
    // property that gating the death handler would destroy.
    arena.dummy.emit('death');
    assert.equal(arena.events.drain().opponent_died, true, 'a post-reset death is still reported');
  },
);

test('handleReset rejects on a throwing chat(), sends no ack, and leaves the damage channel live', async () => {
  const sent = [];
  const arena = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  // A bot that lost its connection between the poll and the reset: chat() throws
  // from the arena:reset_pad call, BEFORE handleReset's try block is entered.
  arena.learner = liveBot('learner_bot', {
    chat: () => {
      throw new Error('cannot chat: bot is not spawned');
    },
  });
  arena.dummy = liveBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.wireDamageEvents();

  await assert.rejects(
    () => arena.handleReset({ type: 'reset', episode: 0, seed: 0 }),
    /cannot chat/,
    'the throw propagates to wireTransport, which reports it as a bridge error',
  );

  assert.deepEqual(sent, [], 'a reset that never ran its gates must not ack');
  assert.equal(
    arena._suppressOpponentEvents,
    false,
    'the flag is raised INSIDE the try, so a throw above it cannot strand suppression on',
  );

  // A stranded flag would zero damage_dealt for the rest of the run — the exact
  // bug class this chain exists to eliminate. The channel must still be live.
  arena.dummy.health = MAX_HEALTH - 6;
  arena.dummy.emit('health');
  assert.equal(arena.events.drain().damage_dealt, 6, 'the damage channel survived the failed reset');
});

// ===========================================================================
// ATTACK-COOLDOWN vs. THE RESET'S REGEAR (T18, issue #28).
//
// THE DEFECT. attack_cooldown reported 1.0 on the first observation of every
// episode because the bridge's only anchor was executor.lastSwingTick, cleared
// to null by the reset. The server's own meter is not necessarily charged
// there, for TWO independent reasons:
//   - Paper 1.21.1's Player.tick() re-zeroes attackStrengthTicker whenever the
//     MAIN-HAND ITEM TYPE differs from the previous tick's, so the regear
//     re-zeroes it when the learner joined empty-handed (air -> iron_sword).
//     A same-tick /clear + /give of the SAME type — every steady-state reset —
//     is invisible to that comparison, so this leg fires on cycle 0 only.
//   - the PREVIOUS episode's final kill swing zeroes it on EVERY cycle
//     (LivingEntity.actuallyHurt resets the attacker's ticker; ServerPlayer
//     .swing does too), and the reset path is not guaranteed to outlast the
//     12.5-tick recovery period — 658-833 ms at this machine's 15-19 TPS.
// Live, the first produced a first swing of 1.269 damage — exactly
// 6 * (0.2 + 0.8 * (1.5/12.5)^2), i.e. ticker == 1 — instead of 6, and told a
// policy it had a full charge it did not have.
//
// THE FIX. handleReset anchors _meterResetTick at tick 0 of the new episode's
// clock and attackCooldown() reports the MINIMUM of the swing ramp and the
// regear ramp, so the value always describes whichever re-zeroing happened
// last. computeAttackCooldown itself is untouched (its four pure tests live in
// transport.test.js) — both ramps go through it.
//
// The anchor is UNCONDITIONAL because a conditional one would be strictly
// worse. It IS derivable (learner.heldItem is readable before the reset goes
// out), but its "accurate" branch — report 1.0 in steady state because the hand
// already holds a sword — is optimistic against the swing-carryover source
// above, which neither heldItem nor death-tracking can observe. The
// unconditional anchor covers swing carryover, the join case and the respawn
// no-op identically. Under-reporting costs a policy at most four IDLE windows;
// over-reporting is the defect. These tests pin that direction on purpose.
// ===========================================================================

/**
 * An EventEmitter bot that satisfies BOTH the reset read-back gate and the step
 * path's snapshot, so one fixture can drive reset -> step -> step. Every added
 * field is one mineflayer really populates on a bot's own connection
 * (`heldItem`) or on its own entity (velocity/yaw/pitch/onGround) — the mock
 * fidelity rule at the top of this file still applies.
 */
function cooldownBot(username, opts = {}) {
  const bot = liveBot(username, opts);
  bot.heldItem = { name: 'iron_sword' };
  Object.assign(bot.entity, {
    username,
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    onGround: true,
  });
  bot.setControlState = () => {};
  bot.clearControlStates = () => {};
  bot.attack = () => {};
  bot.lookAt = () => {};
  return bot;
}

/**
 * A reset-then-step arena: gates satisfied, beacons answered, the REAL
 * MacroExecutor bound to the learner, and the tick wait injected so the step
 * path needs no clock.
 */
function cooldownArena(sent) {
  const arena = new ArenaBots({}, {
    transport: { send: (msg) => sent.push(msg) },
    readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
  });
  arena.learner = cooldownBot('learner_bot');
  arena.dummy = cooldownBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);
  return arena;
}

/** The readiness threshold eval/combat_probe.py applies to the wire value. */
const PROBE_READY = 1.0 - 1e-6;

test('the first observation of an episode reports attack_cooldown 0.0, not a phantom 1.0 (issue #28)', async () => {
  const sent = [];
  const arena = cooldownArena(sent);

  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  assert.equal(sent.length, 2, 'ack + first observation');
  assert.equal(sent[0].ok, true);
  assert.equal(sent[1].type, 'state');
  assert.doesNotThrow(() => validateOutbound(sent[1]));
  // THE REGRESSION. Before T18 this read 1.0 and a policy that attacked here
  // got a partial-cooldown hit while being told the swing was fully charged.
  assert.equal(sent[1].self.attack_cooldown, 0.0);
  assert.ok(
    sent[1].self.attack_cooldown < PROBE_READY,
    'the combat probe must IDLE at w0 rather than swing into an uncharged meter',
  );
  // The wire field itself is untouched: same name, same [0,1] range, same
  // meaning. Only the value is now honest.
  assert.equal(typeof sent[1].self.attack_cooldown, 'number');
});

test('attack_cooldown ramps from the regear over the weapon period and is ready at w4', async () => {
  const sent = [];
  const arena = cooldownArena(sent);
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  const observed = [sent[1].self.attack_cooldown];
  for (let window = 0; window < 4; window += 1) {
    // eslint-disable-next-line no-await-in-loop
    await arena.handleStep({ type: 'step', action: Macro.IDLE });
    observed.push(sent[sent.length - 1].self.attack_cooldown);
  }

  // Derived from the shared period, never hard-coded: ACTION_REPEAT ticks of
  // progress per window against IRON_SWORD_ATTACK_SPEED_TICKS (12.5).
  const expected = [0, 1, 2, 3, 4].map((w) =>
    Math.min((w * ACTION_REPEAT) / IRON_SWORD_ATTACK_SPEED_TICKS, 1.0),
  );
  for (let i = 0; i < expected.length; i += 1) {
    assert.ok(
      Math.abs(observed[i] - expected[i]) < 1e-12,
      `w${i}: expected ${expected[i]}, got ${observed[i]}`,
    );
  }
  // 12.5 ticks is not a multiple of the 4-tick window, so w1..w3 (0.32/0.64/
  // 0.96) all sit BELOW the probe's readiness threshold and only w4 clears it.
  // That ~3.5-tick overshoot is the margin the exact 6,6,6,2 arithmetic rides
  // on — a "corrected" earlier anchor would spend it.
  assert.ok(
    observed.slice(0, 4).every((value) => value < PROBE_READY),
    'w0..w3 must all read not-ready',
  );
  assert.equal(observed[4], 1.0, 'fully charged at w4');
});

test('a swing dominates the reported cooldown once the regear ramp has finished', async () => {
  const sent = [];
  const arena = cooldownArena(sent);
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  // Charge the meter (w1..w4), then swing on the first window that reads ready.
  for (let window = 0; window < 4; window += 1) {
    // eslint-disable-next-line no-await-in-loop
    await arena.handleStep({ type: 'step', action: Macro.IDLE });
  }
  assert.equal(sent[sent.length - 1].self.attack_cooldown, 1.0, 'ready before the swing');

  await arena.handleStep({ type: 'step', action: Macro.ATTACK });

  assert.equal(arena.executor.lastSwingTick, 4 * ACTION_REPEAT, 'the swing stamped the tick');
  // The swing ramp (4 ticks of 12.5) is now behind the regear ramp (20 of 12.5,
  // long since clamped to 1.0), so the MIN must follow the swing.
  assert.ok(
    Math.abs(sent[sent.length - 1].self.attack_cooldown - ACTION_REPEAT / IRON_SWORD_ATTACK_SPEED_TICKS) < 1e-12,
    'the swing re-zeroed the reported meter',
  );
});

test('every episode starts not-ready, not just the first', async () => {
  const sent = [];
  const arena = cooldownArena(sent);
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });
  // Run episode 1 well past the weapon period so the meter is fully charged.
  for (let window = 0; window < 6; window += 1) {
    // eslint-disable-next-line no-await-in-loop
    await arena.handleStep({ type: 'step', action: Macro.IDLE });
  }
  assert.equal(sent[sent.length - 1].self.attack_cooldown, 1.0);

  await arena.handleReset({ type: 'reset', episode: 1, seed: 1 });

  // What this pins is that episode N's first observation reads 0.0 for every N,
  // which catches an anchor that is cleared, moved out of handleReset, or made
  // conditional. It does NOT distinguish one-shot from per-episode arming: the
  // anchor is the constant 0 and every reset re-zeroes _currentTick, so a
  // one-shot mutant is an EQUIVALENT mutant, not a gap this test misses. The
  // distinction only becomes observable if the anchor ever stops being constant.
  assert.equal(sent[sent.length - 1].type, 'state');
  assert.equal(sent[sent.length - 1].self.attack_cooldown, 0.0);
});

test('with no reset and no executor the cooldown still reports 1.0 (the null anchor is inert)', () => {
  // Pre-reset (and for every fake that never runs handleReset) the anchor is
  // null, so attackCooldown() collapses to exactly what it reported before
  // T18. This is what keeps computeAttackCooldown's own no-swing-yet contract
  // — and the transport.test.js cases that pin it — meaningful.
  const arena = new ArenaBots({}, { transport: { send: () => {} } });
  assert.equal(arena._meterResetTick, null);
  assert.equal(arena.attackCooldown(), 1.0);

  arena.executor = new MacroExecutor(cooldownBot('learner_bot'));
  assert.equal(arena.attackCooldown(), 1.0, 'an executor that never swung is charged');
});

// ===========================================================================
// T11b — THE OPPONENT'S ACTION (`opp_action`), AC9.
//
// A second MacroExecutor, bound to the opponent's OWN connection, driven in the
// SAME decision window as the learner's, reporting whether its swing actually
// fired via the optional `state.opp_action_executed`.
//
// THE TWO PROPERTIES THAT MATTER, and why each is tested the way it is:
//
//   1. THE M2 PATH IS BYTE-IDENTICAL. Absent (or null) `opp_action` means the
//      opponent takes no action. That is asserted three ways at once — no
//      executor is ever CREATED, the dummy records not one call, and the
//      encoded state line contains no `opp_action_executed` — because any one
//      of them alone would pass under a mutant that merely looks quiet.
//
//   2. ONE STEP == ONE WINDOW == ACTION_REPEAT GATE TICKS, for the opponent
//      exactly as for the learner. Python cannot see the bridge's `_currentTick`
//      (state.tick is the learner's coarse server world age, flat for several
//      windows then +20), so it reconstructs the opponent's swing meter by
//      COUNTING DECISION WINDOWS. If this side ever advances the opponent's gate
//      by a different amount, or fires twice in one window, the meter desyncs
//      and — per schema.md's swing-report note — the opponent locks into mashing
//      ATTACK every window without ever strafing. The window-index test below is
//      the pin: it asserts the exact window the second swing lands on, derived
//      from the shared constants rather than hard-coded.
// ===========================================================================

/**
 * A cooldownBot that RECORDS everything a macro can do to it, so "the opponent
 * was not driven" is an assertion about calls that did not happen rather than
 * about state that happens to look unchanged.
 */
function drivenBot(username, opts = {}) {
  const bot = cooldownBot(username, opts);
  bot.controlLog = [];
  bot.attacked = [];
  bot.lookedAt = [];
  bot.clearAllCalls = 0;
  bot.setControlState = (state, value) => bot.controlLog.push([state, value]);
  bot.clearControlStates = () => {
    bot.clearAllCalls += 1;
  };
  bot.attack = (entity) => bot.attacked.push(entity);
  bot.lookAt = (point, force) => {
    bot.lookedAt.push({ point, force });
    // Mineflayer's lookAt is async; returning a promise keeps the executor's
    // mandatory .catch on a real code path (an unhandled rejection is
    // process-fatal here and has killed the live bridge before).
    return Promise.resolve();
  };
  return bot;
}

/**
 * A reset-then-step arena whose BOTH bots record what was done to them. Same
 * gate/beacon fixture as cooldownArena — this one just watches the dummy.
 */
function oppArena(sent, errors = []) {
  const arena = new ArenaBots(
    {},
    {
      transport: {
        send: (msg) => sent.push(msg),
        emit: (event, err) => {
          if (event === 'error') errors.push(err);
        },
      },
      readbackOptions: { timeoutMs: 0, pollIntervalMs: 1 },
    },
  );
  arena.learner = drivenBot('learner_bot');
  arena.dummy = drivenBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};
  arena.wireDamageEvents();
  answerResetLikeTheServer(arena);
  return arena;
}

test('TC11: a step with no opp_action drives nothing and leaves the M2 line byte-identical', async () => {
  const sent = [];
  const arena = oppArena(sent);

  await arena.handleStep({ type: 'step', action: Macro.ATTACK });
  await arena.handleStep({ type: 'step', action: Macro.APPROACH });
  // Explicit null is the same statement as an omitted key (schema.md): "the
  // opponent takes no action". It must not be read as macro 0.
  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: null });

  // (a) No second executor was ever constructed, so no code added for the
  //     opponent-acts path can reach the dummy on a dummy-path run at all.
  assert.equal(arena.opponentExecutor, null, 'no opponent executor is created');
  // (b) Nothing was done to the dummy.
  assert.deepEqual(arena.dummy.controlLog, [], 'no control state was pressed on the dummy');
  assert.deepEqual(arena.dummy.attacked, [], 'the dummy swung at nothing');
  assert.deepEqual(arena.dummy.lookedAt, [], 'the dummy turned nowhere');
  assert.equal(arena.dummy.clearAllCalls, 0, 'and its control states were never cleared');
  // (c) The wire is unchanged — asserted on the encoded bytes, not the object.
  assert.equal(sent.length, 3);
  for (const msg of sent) {
    assert.doesNotThrow(() => validateOutbound(msg));
    assert.equal(Object.prototype.hasOwnProperty.call(msg, 'opp_action_executed'), false);
    assert.equal(encodeMessage(msg).includes('opp_action_executed'), false);
  }
  // ...while the LEARNER did act, so this is quiet-because-correct rather than
  // quiet-because-the-step-did-nothing.
  assert.equal(arena.learner.attacked.length, 1, 'the learner still swings');
  assert.equal(arena.learner.attacked[0], arena.dummy.entity);
  assert.deepEqual(arena.learner.controlLog, [
    ['forward', true],
    ['forward', false],
  ]);
});

test('opp_action runs the opponent through the same macro mapping, held for exactly one window', async () => {
  const sent = [];
  const arena = oppArena(sent);

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.APPROACH });

  assert.deepEqual(
    arena.dummy.controlLog,
    [
      ['forward', true],
      ['forward', false],
    ],
    'pressed at the window start and released at its end',
  );
  assert.equal(sent[0].opp_action_executed, false, 'a movement macro is not a swing');
  assert.doesNotThrow(() => validateOutbound(sent[0]));
  // The learner's IDLE holds nothing, and the opponent's macro never leaks onto
  // the learner's connection.
  assert.deepEqual(arena.learner.controlLog, []);
  assert.deepEqual(arena.learner.attacked, []);

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.STRAFE_L });

  assert.deepEqual(arena.dummy.controlLog, [
    ['forward', true],
    ['forward', false],
    ['left', true],
    ['left', false],
  ]);
});

test('opp_action ATTACK swings at the learner and reports opp_action_executed:true', async () => {
  const sent = [];
  const arena = oppArena(sent);

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });

  assert.equal(arena.dummy.attacked.length, 1, 'the opponent swung');
  assert.equal(arena.dummy.attacked[0], arena.learner.entity, 'at the LEARNER');
  assert.equal(sent[0].opp_action_executed, true, 'and the report says the swing went out');
  assert.equal(arena.opponentExecutor.lastSwingTick, 0, 'stamped on the window it began');
  assert.doesNotThrow(() => validateOutbound(sent[0]));
});

test('opp_action TURN_TO_LAST_SEEN faces the learner from a snapshot, not the live vector', async () => {
  const sent = [];
  const arena = oppArena(sent);
  arena.learner.entity.position = livePosition(1.5, 64, 2.5);

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.TURN_TO_LAST_SEEN });

  assert.equal(arena.dummy.lookedAt.length, 1, 'macro 7 must not be a silent no-op for the opponent');
  assert.deepEqual(coordsOf(arena.dummy.lookedAt[0].point), { x: 1.5, y: 64, z: 2.5 });
  assert.equal(arena.dummy.lookedAt[0].force, true, 'force=true bypasses interpolation');
  // A clone, not an alias: the live position keeps moving, the turn target must
  // not. (It is also why lookAt gets a real Vec3 rather than a plain object.)
  arena.learner.entity.position.x = 99;
  assert.equal(arena.dummy.lookedAt[0].point.x, 1.5, 'the turn target was snapshot');
  assert.equal(sent[0].opp_action_executed, false, 'a turn is not a swing');
});

test("the opponent's swing gate advances exactly ACTION_REPEAT per window, in step with the learner's", async () => {
  const sent = [];
  const arena = oppArena(sent);

  // Drive BOTH sides with ATTACK every window. The two gates must agree window
  // for window: Python reconstructs the opponent's meter from the same
  // one-step-one-window count the learner's gate already obeys.
  for (let window = 0; window < 6; window += 1) {
    // eslint-disable-next-line no-await-in-loop
    await arena.handleStep({ type: 'step', action: Macro.ATTACK, opp_action: Macro.ATTACK });
  }

  // Derived from the shared constants, never hard-coded: 12.5 cooldown ticks at
  // ACTION_REPEAT=4 ticks per window means the next swing is allowed at window
  // 4 (elapsed 16 >= 12.5) and NOT at window 3 (12 < 12.5) — the same arithmetic
  // env/mc_pvp_env.py's shadow meter performs as (windows * ACTION_REPEAT) /
  // OPPONENT_ATTACK_SPEED_TICKS.
  const windowsToReady = Math.ceil(IRON_SWORD_ATTACK_SPEED_TICKS / ACTION_REPEAT);
  assert.equal(windowsToReady, 4, 'sanity: the constants still describe a 4-window recharge');

  const reports = sent.map((msg) => msg.opp_action_executed);
  assert.deepEqual(reports, [true, false, false, false, true, false]);
  assert.equal(reports[0], true, 'the first window may swing');
  assert.ok(
    reports.slice(1, windowsToReady).every((r) => r === false),
    'every window inside the cooldown reports a swing that did NOT fire',
  );
  assert.equal(reports[windowsToReady], true, 'and the swing returns on exactly that window');

  assert.equal(arena.dummy.attacked.length, 2, 'two real swings in six windows');
  assert.equal(
    arena.opponentExecutor.lastSwingTick,
    windowsToReady * ACTION_REPEAT,
    'the second swing was stamped with the window-start tick, not the advanced one',
  );
  // The clock invariant itself: the opponent's gate moved exactly as far as the
  // learner's over the same six windows.
  assert.equal(arena.learner.attacked.length, arena.dummy.attacked.length);
  assert.equal(arena.executor.lastSwingTick, arena.opponentExecutor.lastSwingTick);
});

test('the opponent executor is built with the DEFAULT weapon period Python mirrors', async () => {
  const sent = [];
  const arena = oppArena(sent);
  // A learner weapon period that is NOT the default. The opponent's must not
  // follow it: Python hard-codes OPPONENT_ATTACK_SPEED_TICKS = SERVER_TPS / 1.6
  // (env/mc_pvp_env.py) to mirror the DEFAULT, and nothing would catch a drift —
  // the opponent would just quietly stop attacking, or flail.
  arena._weaponAttackSpeedTicks = 99;

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });

  assert.equal(arena.opponentExecutor.weaponAttackSpeedTicks, IRON_SWORD_ATTACK_SPEED_TICKS);
  assert.equal(
    arena.opponentExecutor.weaponAttackSpeedTicks,
    20 / 1.6,
    'SERVER_TPS / 1.6 — the value env/mc_pvp_env.py hard-codes to match',
  );
});

test('an injected opponent executor is used as-is and never rebound mid-run', async () => {
  const sent = [];
  // deps.opponentExecutor mirrors deps.executor: whatever is already bound wins,
  // so a rebind on some later window cannot silently drop the swing gate's
  // accumulated state (and with it, the agreement with Python's shadow meter).
  const dummy = drivenBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  const standIn = new MacroExecutor(dummy);
  const arena = new ArenaBots(
    {},
    { transport: { send: (msg) => sent.push(msg) }, opponentExecutor: standIn },
  );
  arena.learner = drivenBot('learner_bot');
  arena.dummy = dummy;
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};

  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });
  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });

  assert.equal(arena.opponentExecutor, standIn, 'the bound executor is kept');
  assert.equal(standIn.lastSwingTick, 0, 'and it owns the gate across windows');
  assert.deepEqual(
    sent.map((msg) => msg.opp_action_executed),
    [true, false],
  );
});

test('TC23: opp_action with no opponent bot is silently ignored and reported as not executed', async () => {
  // (a) Exhibition: the opponent is a person on their own client and cannot be
  //     puppeted at all.
  const sent = [];
  const challenger = playerEntity('classmate_1', 5.5, 0.5);
  const arena = exhibitionArena(sent, { 11: challenger });
  arena._claimChallenger();

  await assert.doesNotReject(() =>
    arena.handleStep({ type: 'step', action: Macro.ATTACK, opp_action: Macro.ATTACK }),
  );

  assert.equal(sent.length, 1, 'the step still answers with exactly one state');
  assert.doesNotThrow(() => validateOutbound(sent[0]));
  assert.equal(sent[0].opp_action_executed, false, 'nothing executed, and it says so');
  assert.equal(arena.opponentExecutor, null, 'a human is never bound to an executor');
  // The LEARNER's own action is unaffected — AC2 still holds with opp_action set.
  assert.equal(arena.learner.attacked.length, 1);
  assert.equal(arena.learner.attacked[0], challenger);

  // (b) Bot mode with the dummy not connected (pre-spawn, or a failed spawn).
  const botSent = [];
  const botArena = oppArena(botSent);
  botArena.dummy = null;

  await assert.doesNotReject(() =>
    botArena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.APPROACH }),
  );

  assert.equal(botSent.length, 1);
  assert.equal(botSent[0].opp_action_executed, false);
  assert.equal(botArena.opponentExecutor, null);
});

test("every reset re-arms the opponent's swing gate, exactly as it does the learner's", async () => {
  const sent = [];
  const arena = oppArena(sent);
  await arena.handleReset({ type: 'reset', episode: 0, seed: 0 });

  // Swing LATE in episode 0, so the stamp sits far past the next episode's
  // tick 0 and an un-re-armed gate would block the whole opening of episode 1.
  for (let window = 0; window < 6; window += 1) {
    // eslint-disable-next-line no-await-in-loop
    await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.IDLE });
  }
  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });
  assert.equal(arena.opponentExecutor.lastSwingTick, 6 * ACTION_REPEAT, 'a late swing is stamped');
  const clearsBeforeReset = arena.dummy.clearAllCalls;

  await arena.handleReset({ type: 'reset', episode: 1, seed: 1 });

  assert.equal(arena.opponentExecutor.lastSwingTick, null, 'the gate is re-armed');
  assert.ok(
    arena.dummy.clearAllCalls > clearsBeforeReset,
    'and no control state leaks across the episode boundary',
  );

  // Python clears its shadow meter on reset (mc_pvp_env.py sets
  // _opp_last_swing_window = None and reads 1.0), so a bridge that did NOT
  // re-arm here would report false for ~27 windows against a shadow that says
  // ready — the lock-in schema.md warns about.
  await arena.handleStep({ type: 'step', action: Macro.IDLE, opp_action: Macro.ATTACK });
  assert.equal(
    sent[sent.length - 1].opp_action_executed,
    true,
    'the first window of the new episode may swing',
  );
});

// ---------------------------------------------------------------------------
// TC12 — inbound validation ON THE RECEIVE PATH.
//
// `transport.validateInbound` was exported and unit-tested from the start, but
// deliberately never wired into BridgeServer._onData, and handleStep re-checked
// only `action` inline — so an out-of-range `opp_action` was rejected by
// nothing that actually runs. These tests drive the REAL path: raw bytes into a
// socket, through the framer and the 'message' event, into _handleMessage.
//
// The 'error' listener is attached BEFORE the bad frame on purpose: an
// unlistened 'error' on an EventEmitter throws, which is precisely the "bridge
// does not crash" claim under test — without the listener the crash would take
// the assertion with it.
// ---------------------------------------------------------------------------

/** Just enough net.Socket surface for BridgeServer, recording what it writes. */
class RecordingSocket extends EventEmitter {
  constructor() {
    super();
    this.destroyed = false;
    this.written = [];
  }

  setNoDelay() {}

  write(line) {
    this.written.push(line);
    return true;
  }

  destroy() {
    this.destroyed = true;
    this.emit('close');
  }
}

/** Let the fire-and-forget async handler chain settle. */
function flushAsync() {
  return new Promise((resolve) => setImmediate(resolve));
}

/** An arena served by a REAL BridgeServer over a mock socket. */
function receivePathArena() {
  const server = new BridgeServer();
  const errors = [];
  server.on('error', (err) => errors.push(err));
  const arena = new ArenaBots({}, { transport: server });
  arena.learner = drivenBot('learner_bot');
  arena.dummy = drivenBot('dummy_bot', { inventory: [], position: DUMMY_SPAWN });
  arena.executor = new MacroExecutor(arena.learner);
  arena._waitTicksImpl = async () => {};
  arena.wireTransport();
  const socket = new RecordingSocket();
  server._onConnection(socket);
  return { arena, server, socket, errors };
}

test('TC12: an out-of-range opp_action arriving on the wire is a protocol error, not a crash', async () => {
  const { socket, errors } = receivePathArena();

  socket.emit('data', Buffer.from('{"type":"step","action":0,"opp_action":99}\n'));
  await flushAsync();

  assert.equal(errors.length, 1, 'the violation was surfaced exactly once');
  assert.match(errors[0].message, /opp_action/, 'and it names the offending field');
  assert.equal(socket.written.length, 0, 'a rejected step gets no state reply');

  // THE OTHER HALF OF TC12: the bridge is still alive and still serving. A
  // rejected command must not poison the connection or the step path.
  socket.emit('data', Buffer.from('{"type":"step","action":0,"opp_action":3}\n'));
  await flushAsync();

  assert.equal(errors.length, 1, 'the valid step raised nothing');
  assert.equal(socket.written.length, 1, 'and was answered');
  const reply = JSON.parse(socket.written[0]);
  assert.equal(reply.type, 'state');
  assert.equal(reply.opp_action_executed, false, 'STRAFE_L executed, but it is not a swing');
});

test('TC12: every shape the opp_action rule rejects is rejected on the receive path', async () => {
  // Exactly the cases transport.validateStep enumerates — non-integer, boolean
  // (which JS would coerce), string, and both ends of the frozen 0..7 range.
  const bad = ['99', '-1', '1.5', 'true', '"3"'];
  for (const value of bad) {
    const { socket, errors } = receivePathArena();
    socket.emit('data', Buffer.from(`{"type":"step","action":0,"opp_action":${value}}\n`));
    // eslint-disable-next-line no-await-in-loop
    await flushAsync();
    assert.equal(errors.length, 1, `opp_action ${value} must be rejected`);
    assert.match(errors[0].message, /opp_action/);
    assert.equal(socket.written.length, 0, `opp_action ${value} must not be executed`);
  }

  // ...and the M2 line still flows through the same guard untouched.
  const { socket, errors } = receivePathArena();
  socket.emit('data', Buffer.from('{"type":"step","action":5}\n'));
  await flushAsync();
  assert.deepEqual(errors, []);
  assert.equal(socket.written.length, 1);
  assert.equal(
    socket.written[0].includes('opp_action_executed'),
    false,
    'no opp_action in, no swing report out',
  );
});

test('the inline action guard it replaced still behaves the same: bad action, one error, no state', async () => {
  const { socket, errors } = receivePathArena();

  socket.emit('data', Buffer.from('{"type":"step","action":99}\n'));
  await flushAsync();

  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /action/);
  assert.equal(socket.written.length, 0);
});

// ===========================================================================
// THE IN-GAME CHAT RESET — demo day's "no alt-tab" trigger.
//
// A player types `reset` in Minecraft chat and the bridge files the SAME
// request file `python -m deploy.exhibition --reset` files; the launcher's
// existing poll does the rest. Nothing here resets anything — that is
// deliberate, and it is why these tests assert on ONE file write and one line
// of chat rather than on arena state.
//
// FOUR BEHAVIORS ARE LOAD-BEARING AND EACH HAS ITS OWN TEST, because each one
// fails in a way an operator standing in front of a room cannot debug:
//   - the EXHIBITION GATE. Training is 25 pads in one JVM all hearing the same
//     chat; a stray keyword must never perturb a run.
//   - the OWN-BOT FILTER. The server echoes chat back to the sender, so the
//     bridge hears its own confirmation on the same channel it triggers from.
//   - WHOLE-MESSAGE matching. "how do I reset?" must not end a live match.
//   - the DEBOUNCE. A queue of people all typing `reset` at once is one match,
//     not five.
//
// MOCK FIDELITY: the file writes here are REAL, into a real temp dir, through
// the real fs. The write is the entire mechanism, and a stubbed fs would prove
// only that a stub was called. The failure case uses a real ENOENT (a path
// under a directory that does not exist) for the same reason.
// ===========================================================================

/**
 * A temp dir that cleans itself up when the process exits.
 *
 * ONE `exit` listener for all of them, not one each: node warns about a
 * possible leak past ten listeners on the same emitter, and a suite that prints
 * a MaxListenersExceededWarning teaches everyone to ignore its warnings.
 */
const TEMP_LOG_DIRS = [];
process.on('exit', () => {
  for (const dir of TEMP_LOG_DIRS) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

function tempLogDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mc-chat-reset-'));
  TEMP_LOG_DIRS.push(dir);
  return dir;
}

/** A clock the debounce tests can move without spending wall-clock in it. */
function fakeClock(start = 1_700_000_000_000) {
  const state = { ms: start };
  return {
    now: () => state.ms,
    advance(ms) {
      state.ms += ms;
    },
  };
}

/**
 * An arena wired exactly as connect() wires one, with a chat-capable learner.
 *
 * The usernames are DELIBERATELY not `learner_bot`/`dummy_bot`: the own-bot
 * filter has to compare against the resolved, configured names (pad 4 runs
 * `learner_4`, and the exhibition can rename either), so a fixture using the
 * defaults would let a filter hard-coded to the literals pass.
 */
function chatArena({
  requestPath = null,
  opponentMode = OPPONENT_MODE_HUMAN,
  now = Date.now,
  chat,
} = {}) {
  const arena = new ArenaBots(
    {
      opponentMode,
      resetRequestPath: requestPath,
      learnerUsername: 'learner_9',
      dummyUsername: 'dummy_9',
    },
    { transport: { send: () => {} }, nowMs: now },
  );
  arena.learner = liveBot('learner_9', { chat });
  // 'bot' mode is the training path, and there the dummy really is connected.
  if (opponentMode !== OPPONENT_MODE_HUMAN) {
    arena.dummy = liveBot('dummy_9');
  }
  arena.wireDamageEvents();
  return arena;
}

/**
 * One player says one line, with console.error captured.
 *
 * Captured rather than silenced: "it was logged" is half of what a
 * filesystem failure must do (the other half is not taking the bridge down),
 * and the capture keeps the suite's own output readable.
 *
 * @returns {string[]} Whatever the handler logged.
 */
function say(arena, username, message) {
  const logged = [];
  const realError = console.error;
  console.error = (...args) => logged.push(args.map((a) => String(a)).join(' '));
  try {
    arena.learner.emit('chat', username, message);
  } finally {
    console.error = realError;
  }
  return logged;
}

test('the keyword is the WHOLE message: `reset` and `!reset`, any case, trimmed', () => {
  for (const typed of ['reset', '!reset', 'RESET', '!ReSeT', '  reset  ', '\treset\n', ' !RESET ']) {
    assert.equal(matchesChatResetKeyword(typed), true, `${JSON.stringify(typed)} must trigger`);
  }

  // THE DANGEROUS DIRECTION. Chat during a demo is full of sentences with the
  // word in them, and a substring match would end a live match mid-fight on
  // any of these.
  for (const typed of [
    'how do I reset?',
    'can we reset after this one',
    'reset now',
    'please reset',
    'resets',
    'preset',
    '!resetting',
    '/reset',
    '',
    '   ',
  ]) {
    assert.equal(matchesChatResetKeyword(typed), false, `${JSON.stringify(typed)} must NOT trigger`);
  }

  // Nothing that is not a string is a keyword — the handler is fed whatever
  // the chat plugin's regex produced.
  for (const junk of [null, undefined, 42, {}, ['reset']]) {
    assert.equal(matchesChatResetKeyword(junk), false);
  }
});

test('the confirmation the bridge says back is not itself a keyword', () => {
  // The server echoes chat to the sender, so the confirmation returns on the
  // very channel that triggers a reset. The own-bot filter is what stops the
  // loop; this is the second lock on the same door.
  assert.equal(matchesChatResetKeyword(CHAT_RESET_CONFIRMATION), false);
});

test('a player typing `reset` files the request and is told so in game', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const clock = fakeClock();
  const arena = chatArena({ requestPath, now: clock.now });

  const logged = say(arena, 'classmate_1', 'reset');

  // THE MECHANISM: the file the launcher polls, at the path it was given.
  assert.equal(fs.existsSync(requestPath), true, 'the reset request must be on disk');
  assert.equal(
    fs.readFileSync(requestPath, 'utf8'),
    formatChatResetRequest('classmate_1', new Date(clock.now()).toISOString()),
  );
  // THE UX: a silent trigger gets typed five more times by someone who cannot
  // see a terminal.
  assert.deepEqual(arena.learner.chatLog, [CHAT_RESET_CONFIRMATION]);
  assert.equal(
    logged.some((line) => line.includes('classmate_1') && line.includes(requestPath)),
    true,
    'bridge.log must name who armed it and where',
  );
});

test('`!reset`, odd casing and stray whitespace all arm it too', () => {
  for (const typed of ['!reset', 'RESET', '  !ReSeT  ']) {
    const requestPath = path.join(tempLogDir(), 'reset.request');
    const arena = chatArena({ requestPath });

    say(arena, 'classmate_1', typed);

    assert.equal(fs.existsSync(requestPath), true, `${JSON.stringify(typed)} must arm a reset`);
  }
});

test('a sentence that merely contains the word arms nothing', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const arena = chatArena({ requestPath });

  say(arena, 'classmate_1', 'how do I reset?');
  say(arena, 'classmate_1', 'reset now');

  assert.equal(fs.existsSync(requestPath), false, 'a live match must not end on a question');
  assert.deepEqual(arena.learner.chatLog, []);
});

test('our own bots are not players: their chat is ignored, by resolved name', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const arena = chatArena({ requestPath });

  // Both of THIS pad's bots, under their configured names — not the
  // learner_bot/dummy_bot literals. A bot reacting to its own line would loop.
  say(arena, 'learner_9', 'reset');
  say(arena, 'dummy_9', 'reset');

  assert.equal(fs.existsSync(requestPath), false, 'own-bot chat must never arm a reset');
  assert.deepEqual(arena.learner.chatLog, []);

  // ...and a real person is still heard on the same channel.
  say(arena, 'classmate_1', 'reset');
  assert.equal(fs.existsSync(requestPath), true);
});

test('a burst from a queue of classmates arms exactly ONE reset', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const clock = fakeClock();
  const arena = chatArena({ requestPath, now: clock.now });

  say(arena, 'classmate_1', 'reset');
  const armedAt = fs.readFileSync(requestPath, 'utf8');

  // Four more people (and one of them twice) inside the cooldown.
  clock.advance(200);
  say(arena, 'classmate_2', 'reset');
  clock.advance(900);
  say(arena, 'classmate_3', '!reset');
  clock.advance(1500);
  say(arena, 'classmate_3', 'reset');
  clock.advance(CHAT_RESET_DEBOUNCE_MS - 2601); // still inside the window
  say(arena, 'classmate_4', 'RESET');

  // One file (untouched since the first write) and ONE line of chat: a burst
  // of confirmations is spam in front of a room, and a burst of writes would
  // race the launcher's consume.
  assert.equal(fs.readFileSync(requestPath, 'utf8'), armedAt);
  assert.deepEqual(arena.learner.chatLog, [CHAT_RESET_CONFIRMATION]);
});

test('the cooldown expires, so the NEXT challenger can arm their own match', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const clock = fakeClock();
  const arena = chatArena({ requestPath, now: clock.now });

  say(arena, 'classmate_1', 'reset');
  fs.unlinkSync(requestPath); // the launcher consumed it and played a match

  clock.advance(CHAT_RESET_DEBOUNCE_MS);

  say(arena, 'classmate_2', 'reset');

  // A debounce that never released would give an evening exactly one reset.
  assert.equal(fs.existsSync(requestPath), true);
  assert.deepEqual(arena.learner.chatLog, [CHAT_RESET_CONFIRMATION, CHAT_RESET_CONFIRMATION]);
});

test('TRAINING IS INERT: in `bot` opponent mode the keyword does nothing', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  // A path is configured — the gate under test is the OPPONENT MODE, not the
  // absence of a path. 25 training arenas all read chat, and a classmate who
  // wanders onto the training server must not perturb an overnight run.
  const arena = chatArena({ requestPath, opponentMode: 'bot' });

  say(arena, 'classmate_1', 'reset');
  say(arena, 'classmate_1', '!reset');

  assert.equal(fs.existsSync(requestPath), false, 'a training run must not see a reset request');
  assert.deepEqual(arena.learner.chatLog, []);
});

test('with no request path configured the feature does not exist', () => {
  const arena = chatArena({ requestPath: null });

  assert.equal(arena.resetRequestPath, null);
  // Nothing to write to, so nothing is written and nothing is promised.
  assert.doesNotThrow(() => say(arena, 'classmate_1', 'reset'));
  assert.deepEqual(arena.learner.chatLog, []);
});

test('a write failure is logged, says nothing in chat, and never escapes', () => {
  // A REAL ENOENT: the parent directory does not exist. The bridge is serving
  // the RL wire and an exception out of a mineflayer listener is process-fatal
  // here, so this must cost the reset and nothing else.
  const requestPath = path.join(tempLogDir(), 'no', 'such', 'dir', 'reset.request');
  const clock = fakeClock();
  const arena = chatArena({ requestPath, now: clock.now });

  let logged;
  assert.doesNotThrow(() => {
    logged = say(arena, 'classmate_1', 'reset');
  });

  assert.equal(fs.existsSync(requestPath), false);
  assert.equal(
    logged.some((line) => line.includes('could not file the request')),
    true,
    'a filesystem failure must be loud in bridge.log',
  );
  // NO "reset armed" for a request that was never filed — the one lie this
  // handler must not tell.
  assert.deepEqual(arena.learner.chatLog, []);
  // ...and the failure did not arm the cooldown, so the next attempt is fresh
  // rather than swallowed for 5 s on the strength of a write that never was.
  assert.equal(arena._lastChatResetMs, null);
});

test('a throwing bot.chat() cannot take the bridge down after the request is filed', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const arena = chatArena({
    requestPath,
    chat: () => {
      throw new Error('client disconnected');
    },
  });

  assert.doesNotThrow(() => say(arena, 'classmate_1', 'reset'));

  // The reset is what matters and it is already on disk; the confirmation is
  // cosmetic. Losing the bridge here would end the exhibition.
  assert.equal(fs.existsSync(requestPath), true);
});

test('a rejected bot.chat() promise is caught, not left to kill the process', async () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const arena = chatArena({
    requestPath,
    chat: () => Promise.reject(new Error('chat rejected')),
  });

  const rejections = [];
  const onUnhandled = (reason) => rejections.push(reason);
  process.on('unhandledRejection', onUnhandled);
  try {
    say(arena, 'classmate_1', 'reset');
    // Unhandled rejections are reported a turn later, so let the microtask
    // queue drain before asserting there were none.
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.off('unhandledRejection', onUnhandled);
  }

  assert.deepEqual(rejections, [], 'a fire-and-forget bot promise is process-fatal here');
  assert.equal(fs.existsSync(requestPath), true);
});

test('re-wiring does not double-register the chat handler', () => {
  const requestPath = path.join(tempLogDir(), 'reset.request');
  const clock = fakeClock();
  const arena = chatArena({ requestPath, now: clock.now });

  // A reconnect re-wires. The handler must be removed from where it was added
  // (W1a), or a relaunched bridge answers one keyword twice and files two
  // requests — the second of which would arm a match nobody asked for.
  arena.wireDamageEvents();
  arena.wireDamageEvents();

  assert.equal(arena.learner.listenerCount('chat'), 1);

  say(arena, 'classmate_1', 'reset');
  assert.deepEqual(arena.learner.chatLog, [CHAT_RESET_CONFIRMATION]);
});
