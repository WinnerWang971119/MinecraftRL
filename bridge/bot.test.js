// bot.test.js — `node --test` suite for the reset regear path. Runs WITHOUT a
// live Minecraft server.
//
// ============================================================================
// WHAT THIS FILE VERIFIES (testable now, no Paper server):
//   ArenaBots._regear (bot.js — the go-live Step 3 gate failure):
//     - issues /clear FIRST, then one namespaced /give per template item, all
//       through the bot's own (opped) chat;
//     - is a safe no-op for a bot with no username yet (pre-login poll).
//   ArenaBots.handleReset (mock bots + mock transport):
//     - regears BOTH bots and replies reset_ack ok:true when BOTH read-back
//       gates match their templates INCLUDING the given gear — the exact
//       configuration that failed live before _regear was implemented;
//     - a STALE reset that loses the epoch race applies none of its four
//       post-gate effects, so a retry's live episode survives untouched;
//     - a dummy `death` fired during the reset window is discarded by the
//       winning handler's events.reset(), NOT by the suppression flag;
//     - a throwing bot.chat() rejects without acking and without stranding
//       the suppression flag, leaving the damage channel live.
//
// MOCK FIDELITY — READ BEFORE ADDING A FAKE HERE.
//   Mineflayer populates `health` ONLY on a bot's own connection; the entity
//   view of another player never carries it (prismarine-entity defines no such
//   field). The fakes below therefore put `health` on the BOT and never on
//   `entity`. A fake more capable than the real library is how the damage
//   channel shipped dead — see bridge/actions.test.js for the full note.
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (server/compat_check.md):
//   The real /give landing in a real inventory within the gate's 3 s timeout,
//   and the dummy's real post-heal health landing within the dummy gate.
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');

const { ArenaBots, MAX_HEALTH, ACTION_REPEAT } = require('./bot');
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
 * spawn, holding the template gear, no active effects. Without this the dummy
 * gate legitimately rejects the fake and burns its full 3 s timeout.
 */
function mockDummy(overrides = {}) {
  return mockBot('dummy_bot', {
    inventory: ['iron_sword'],
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

test('_regear issues /clear first, then a namespaced /give per template item', () => {
  const bots = new ArenaBots({}, { transport: { send: () => {} } });
  const bot = mockBot('learner_bot');

  bots._regear(bot);

  assert.deepEqual(bot.chatLog, [
    '/clear learner_bot',
    '/give learner_bot minecraft:iron_sword 1',
  ]);
});

test('_regear is a no-op for a bot without a username (pre-login)', () => {
  const bots = new ArenaBots({}, { transport: { send: () => {} } });
  const bot = mockBot(undefined);

  bots._regear(bot);

  assert.deepEqual(bot.chatLog, []);
});

test('handleReset regears both bots, acks ok:true, then sends the initial state', async () => {
  const sent = [];
  const bots = new ArenaBots({}, { transport: { send: (msg) => sent.push(msg) } });
  // BOTH bots already read back their templates (live, the /tp, /effect and
  // /give land during the gates' poll window; mocks have no async command
  // latency). The dummy gate is as load-bearing as the learner's: acking while
  // the dummy is still hurt would measure the first real hit against a phantom
  // baseline, so its mock must sit at the +3 x spawn, healed and regeared.
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
  // Both bots got the forced heal and the full regear, alongside their /tp
  // and /effect clear.
  for (const bot of [bots.learner, bots.dummy]) {
    assert.ok(bot.chatLog.includes(`/effect give ${bot.username} minecraft:instant_health 1 10 true`));
    assert.ok(bot.chatLog.includes(`/clear ${bot.username}`));
    assert.ok(bot.chatLog.includes(`/give ${bot.username} minecraft:iron_sword 1`));
    assert.ok(bot.chatLog.indexOf(`/clear ${bot.username}`)
      < bot.chatLog.indexOf(`/give ${bot.username} minecraft:iron_sword 1`));
  }
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
  arena.dummy = liveBot('dummy_bot', { position: DUMMY_SPAWN });
  arena.dummy.health = 18;
  arena.wireDamageEvents();
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
  arena.dummy = liveBot('dummy_bot', { position: DUMMY_SPAWN });
  arena.wireDamageEvents();

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
    arena.dummy = liveBot('dummy_bot', { position: DUMMY_SPAWN });
    arena.wireDamageEvents();
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
  // from the very first /tp, BEFORE handleReset's try block is entered.
  arena.learner = liveBot('learner_bot', {
    chat: () => {
      throw new Error('cannot chat: bot is not spawned');
    },
  });
  arena.dummy = liveBot('dummy_bot', { position: DUMMY_SPAWN });
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
