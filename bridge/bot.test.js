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
const path = require('node:path');

const {
  ArenaBots,
  MAX_HEALTH,
  ACTION_REPEAT,
  formatResetPadCommand,
  formatSetupPadCommand,
} = require('./bot');
const { validateOutbound } = require('./transport');
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
