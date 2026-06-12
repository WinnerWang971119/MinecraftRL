// actions.test.js — `node --test` suite for the T7b macro execution + damage
// event aggregation. Runs WITHOUT a live Minecraft server.
//
// ============================================================================
// WHAT THIS FILE VERIFIES (testable now, no Paper server):
//   EventAggregator (actions.js — THE deepest silent risk):
//     - TC7 a 3-hit exchange straddling a window boundary sums to EXACTLY 3 — no
//       drop, no double-count at the boundary;
//     - hits at the FIRST tick of a window, the LAST tick, and split across two
//       windows are each counted once;
//     - damage_dealt / damage_taken accumulate; negatives / non-finite ignored;
//     - i_died / opponent_died LATCH once per window and reset on drain;
//     - reset() discards a partial (un-drained) window.
//   Macro -> control-state mapping (actions.js):
//     - each movement macro sets the right control state(s); IDLE sets none;
//     - the mapping references NO pathfinder goal and NO bot.pvp.
//   MacroExecutor (actions.js):
//     - begin() sets the correct control states; end() clears exactly those;
//     - ATTACK calls bot.attack(entity) (a single swing) and respects the
//       cooldown gate — a second ATTACK within the weapon cooldown does NOT
//       swing again; it swings again once the cooldown elapses;
//     - ATTACK / movement NEVER touch bot.pvp.attack or a pathfinder goal;
//     - TURN_TO_LAST_SEEN calls bot.lookAt toward the stored last-seen position;
//       with no memory it is a no-op.
//   State assembly (bot.js buildEventsBlock / assembleStateMsg):
//     - the assembled `state` is schema-valid (transport.validateOutbound), and
//       carries the drained events block;
//     - the aggregator drain feeds straight into a valid events block.
//   ArenaBots.handleStep (bot.js — wired with an injected tick wait + mock bots):
//     - one step runs the macro, drains exactly the window's events, and sends
//       one schema-valid `state`; damage that arrives during the window is
//       counted exactly once and is cleared from the next window.
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (NOT covered here — server/compat_check.md):
//   TC7b the real damage exchange: two opped bots, real entityHurt timing, real
//        swing cooldown actually moving health. A documented human follow-up.
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  Macro,
  N_ACTIONS,
  ACTION_MIN,
  ACTION_MAX,
  macroToControlStates,
  isAttackMacro,
  isTurnMacro,
  EventAggregator,
  MacroExecutor,
} = require('./actions');

const { validateOutbound } = require('./transport');
const { buildEventsBlock, assembleStateMsg, ArenaBots, ACTION_REPEAT } = require('./bot');

// ===========================================================================
// Frozen enum sanity — must mirror agent/actions.py Macro IntEnum exactly.
// ===========================================================================

test('Macro enum mirrors the frozen agent/actions.py values 0..7', () => {
  assert.equal(Macro.IDLE, 0);
  assert.equal(Macro.APPROACH, 1);
  assert.equal(Macro.RETREAT, 2);
  assert.equal(Macro.STRAFE_L, 3);
  assert.equal(Macro.STRAFE_R, 4);
  assert.equal(Macro.ATTACK, 5);
  assert.equal(Macro.JUMP, 6);
  assert.equal(Macro.TURN_TO_LAST_SEEN, 7);
  assert.equal(N_ACTIONS, 8);
  assert.equal(ACTION_MIN, 0);
  assert.equal(ACTION_MAX, 7);
  assert.equal(ACTION_REPEAT, 4, 'frozen frame-skip from agent/contract_config.py');
});

// ===========================================================================
// EventAggregator — TC7: exactly-once counting at the window boundary.
//
// "Per-tick" events are fed via recordX; "the window boundary" is defined by
// WHEN drain() is called. The straddle test feeds hits across a drain() and
// asserts the totals across the two windows sum to exactly 3 — no drop, no
// double-count.
// ===========================================================================

test('TC7 a 3-hit exchange straddling a window boundary sums to EXACTLY 3', () => {
  const agg = new EventAggregator();

  // Window 1: two hits land before the boundary.
  agg.recordDamageDealt(1); // hit #1
  agg.recordDamageDealt(1); // hit #2
  const w1 = agg.drain(); // <-- window boundary

  // Window 2: the third hit lands after the boundary.
  agg.recordDamageDealt(1); // hit #3
  const w2 = agg.drain();

  assert.equal(w1.damage_dealt, 2, 'window 1 owns exactly the two hits before the boundary');
  assert.equal(w2.damage_dealt, 1, 'window 2 owns exactly the one hit after the boundary');
  assert.equal(
    w1.damage_dealt + w2.damage_dealt,
    3,
    'the 3-hit exchange sums to exactly 3 across the boundary — no drop, no double-count',
  );
});

test('TC7 a hit at the FIRST tick of a window is counted once, in that window', () => {
  const agg = new EventAggregator();
  // Open a fresh window with a drain, then the very first thing in the new
  // window is a hit.
  agg.drain();
  agg.recordDamageDealt(1); // first-tick hit
  agg.recordDamageDealt(1);
  const w = agg.drain();
  assert.equal(w.damage_dealt, 2, 'first-tick hit is included exactly once');
  const next = agg.drain();
  assert.equal(next.damage_dealt, 0, 'the next window does NOT re-see the first-tick hit');
});

test('TC7 a hit at the LAST tick of a window is counted once, not leaked forward', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(1);
  agg.recordDamageDealt(1);
  agg.recordDamageDealt(1); // last-tick hit, just before the boundary
  const w = agg.drain(); // boundary immediately after the last-tick hit
  assert.equal(w.damage_dealt, 3, 'last-tick hit is in THIS window');
  const next = agg.drain();
  assert.equal(next.damage_dealt, 0, 'last-tick hit does NOT leak into the next window');
});

test('TC7 hits split across two windows are each counted once (no boundary drop)', () => {
  const agg = new EventAggregator();
  // 3 hits in window A, 2 hits in window B — totals must be 3 and 2, sum 5.
  agg.recordDamageDealt(1);
  agg.recordDamageDealt(1);
  agg.recordDamageDealt(1);
  const a = agg.drain();
  agg.recordDamageDealt(1);
  agg.recordDamageDealt(1);
  const b = agg.drain();
  assert.equal(a.damage_dealt, 3);
  assert.equal(b.damage_dealt, 2);
  assert.equal(a.damage_dealt + b.damage_dealt, 5, 'no hit dropped at the boundary');
});

test('EventAggregator accumulates damage_dealt and damage_taken independently', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(3);
  agg.recordDamageDealt(2.5);
  agg.recordDamageTaken(4);
  agg.recordDamageTaken(1);
  const w = agg.drain();
  assert.equal(w.damage_dealt, 5.5);
  assert.equal(w.damage_taken, 5);
});

test('EventAggregator ignores non-positive and non-finite damage (defensive)', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(0); // a no-op hit
  agg.recordDamageDealt(-3); // a heal must not register as negative damage
  agg.recordDamageDealt(NaN);
  agg.recordDamageDealt(Infinity);
  agg.recordDamageDealt('5'); // wrong type
  agg.recordDamageTaken(-1);
  const w = agg.drain();
  assert.equal(w.damage_dealt, 0, 'no spurious damage from non-positive / non-finite input');
  assert.equal(w.damage_taken, 0);
});

test('EventAggregator i_died / opponent_died LATCH once and reset on drain', () => {
  const agg = new EventAggregator();
  assert.equal(agg.peek().i_died, false);
  assert.equal(agg.peek().opponent_died, false);

  // Repeated death signals within one window collapse to a single true.
  agg.recordIDied();
  agg.recordIDied();
  agg.recordOpponentDied();
  const w = agg.drain();
  assert.equal(w.i_died, true, 'i_died latched');
  assert.equal(w.opponent_died, true, 'opponent_died latched');

  // The death belongs to exactly the window it happened in: the next window
  // starts clean.
  const next = agg.drain();
  assert.equal(next.i_died, false, 'death does not carry into the next window');
  assert.equal(next.opponent_died, false);
});

test('EventAggregator peek() does not clear; drain() does', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(2);
  assert.equal(agg.peek().damage_dealt, 2, 'peek reports the pending total');
  assert.equal(agg.peek().damage_dealt, 2, 'peek is idempotent (does not clear)');
  assert.equal(agg.drain().damage_dealt, 2, 'drain reports it once');
  assert.equal(agg.drain().damage_dealt, 0, 'drain cleared it (no double-count)');
});

test('EventAggregator reset() discards a partial, un-drained window', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(7);
  agg.recordIDied();
  agg.reset(); // e.g. a new episode mid-window
  const w = agg.drain();
  assert.equal(w.damage_dealt, 0, 'reset threw away the partial window');
  assert.equal(w.i_died, false);
});

// ===========================================================================
// Macro -> control-state mapping — each movement macro maps to the right key(s);
// IDLE/ATTACK/TURN map to none; the mapping is pathfinder-free.
// ===========================================================================

test('macroToControlStates maps each movement macro to the correct control state', () => {
  assert.deepEqual(macroToControlStates(Macro.APPROACH), ['forward']);
  assert.deepEqual(macroToControlStates(Macro.RETREAT), ['back']);
  assert.deepEqual(macroToControlStates(Macro.STRAFE_L), ['left']);
  assert.deepEqual(macroToControlStates(Macro.STRAFE_R), ['right']);
  assert.deepEqual(macroToControlStates(Macro.JUMP), ['jump']);
});

test('macroToControlStates maps IDLE / ATTACK / TURN_TO_LAST_SEEN to NO control state', () => {
  assert.deepEqual(macroToControlStates(Macro.IDLE), []);
  assert.deepEqual(macroToControlStates(Macro.ATTACK), []);
  assert.deepEqual(macroToControlStates(Macro.TURN_TO_LAST_SEEN), []);
});

test('macroToControlStates throws on an out-of-range macro index', () => {
  assert.throws(() => macroToControlStates(8), RangeError);
  assert.throws(() => macroToControlStates(-1), RangeError);
});

test('isAttackMacro / isTurnMacro identify the special (non-control-state) macros', () => {
  assert.equal(isAttackMacro(Macro.ATTACK), true);
  assert.equal(isAttackMacro(Macro.APPROACH), false);
  assert.equal(isTurnMacro(Macro.TURN_TO_LAST_SEEN), true);
  assert.equal(isTurnMacro(Macro.ATTACK), false);
});

// ===========================================================================
// MacroExecutor — control states pressed on begin and released on end; the
// cooldown-gated single swing; pvp/pathfinder are never used.
// ===========================================================================

/**
 * A recording mock Mineflayer bot. Tracks control states, attack/lookAt calls,
 * AND exposes a pvp + pathfinder surface whose use we assert NEVER happens (if
 * the executor wrongly reached for them, the recordings would be non-empty).
 */
function makeRecordingBot() {
  const bot = {
    controlStates: {},
    attackCalls: [],
    lookAtCalls: [],
    clearAllCount: 0,
    pvpAttackCalls: [],
    pathfinderGoals: [],
    setControlState(name, value) {
      bot.controlStates[name] = value;
    },
    clearControlStates() {
      bot.clearAllCount += 1;
      for (const k of Object.keys(bot.controlStates)) {
        bot.controlStates[k] = false;
      }
    },
    attack(entity) {
      bot.attackCalls.push(entity);
    },
    lookAt(pos, force) {
      bot.lookAtCalls.push({ pos, force });
    },
    pvp: {
      attack(entity) {
        bot.pvpAttackCalls.push(entity);
      },
    },
    pathfinder: {
      setGoal(goal) {
        bot.pathfinderGoals.push(goal);
      },
    },
  };
  return bot;
}

test('MacroExecutor.begin sets the movement control state and end() clears exactly it', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);

  exec.begin(Macro.APPROACH, { currentTick: 0 });
  assert.equal(bot.controlStates.forward, true, 'APPROACH pressed forward');
  assert.equal(bot.controlStates.back, undefined, 'no other key pressed');

  exec.end();
  assert.equal(bot.controlStates.forward, false, 'end() released forward');
});

test('MacroExecutor IDLE presses no control state', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);
  exec.begin(Macro.IDLE, { currentTick: 0 });
  assert.deepEqual(bot.controlStates, {}, 'IDLE touched no control state');
  assert.equal(bot.attackCalls.length, 0, 'IDLE does not attack');
  exec.end();
});

test('MacroExecutor releases the previous macro key when a new macro is begun', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);
  exec.begin(Macro.APPROACH, { currentTick: 0 });
  assert.equal(bot.controlStates.forward, true);
  exec.end();
  exec.begin(Macro.STRAFE_L, { currentTick: 4 });
  assert.equal(bot.controlStates.forward, false, 'forward not still held under STRAFE_L');
  assert.equal(bot.controlStates.left, true, 'STRAFE_L pressed left');
  exec.end();
});

test('MacroExecutor ATTACK calls bot.attack(entity) for a single swing', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot, { weaponAttackSpeedTicks: 10 });
  const opponent = { id: 'dummy-entity' };

  const r = exec.begin(Macro.ATTACK, { currentTick: 0, opponentEntity: opponent });
  assert.equal(r.swung, true, 'a swing happened');
  assert.equal(bot.attackCalls.length, 1, 'exactly one bot.attack swing');
  assert.equal(bot.attackCalls[0], opponent, 'swung at the opponent entity');
});

test('MacroExecutor ATTACK is cooldown-gated: a second ATTACK within cooldown does NOT swing', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot, { weaponAttackSpeedTicks: 10 });
  const opponent = { id: 'dummy-entity' };

  // First swing at tick 0.
  const first = exec.begin(Macro.ATTACK, { currentTick: 0, opponentEntity: opponent });
  assert.equal(first.swung, true);
  assert.equal(bot.attackCalls.length, 1);

  // Second ATTACK only 4 ticks later — inside the 10-tick weapon cooldown.
  const second = exec.begin(Macro.ATTACK, { currentTick: 4, opponentEntity: opponent });
  assert.equal(second.swung, false, 'still cooling down — no swing');
  assert.equal(bot.attackCalls.length, 1, 'no second bot.attack call within cooldown');

  // Once the full cooldown has elapsed, the next ATTACK swings again.
  const third = exec.begin(Macro.ATTACK, { currentTick: 10, opponentEntity: opponent });
  assert.equal(third.swung, true, 'cooldown elapsed — swings again');
  assert.equal(bot.attackCalls.length, 2);
});

test('MacroExecutor ATTACK with no opponent entity does not swing and does not start cooldown', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot, { weaponAttackSpeedTicks: 10 });
  const r = exec.begin(Macro.ATTACK, { currentTick: 0, opponentEntity: null });
  assert.equal(r.swung, false, 'no target -> no swing');
  assert.equal(bot.attackCalls.length, 0);
  assert.equal(exec.lastSwingTick, null, 'no phantom cooldown started');

  // A real target on the very next tick must still be allowed to swing.
  const r2 = exec.begin(Macro.ATTACK, { currentTick: 1, opponentEntity: { id: 'x' } });
  assert.equal(r2.swung, true, 'a swing-at-nothing did not consume the cooldown');
});

test('MacroExecutor resetCooldown re-arms the swing gate for a new episode', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot, { weaponAttackSpeedTicks: 10 });
  exec.begin(Macro.ATTACK, { currentTick: 0, opponentEntity: { id: 'x' } });
  assert.equal(exec.lastSwingTick, 0);
  exec.resetCooldown();
  assert.equal(exec.lastSwingTick, null, 'cooldown re-armed');
  // After reset, an immediate swing at tick 1 is allowed (fresh episode).
  const r = exec.begin(Macro.ATTACK, { currentTick: 1, opponentEntity: { id: 'x' } });
  assert.equal(r.swung, true);
});

test('MacroExecutor never uses bot.pvp.attack or a pathfinder goal for ATTACK/movement', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot, { weaponAttackSpeedTicks: 5 });
  const opponent = { id: 'dummy-entity' };

  // Exercise the attack + every movement macro.
  exec.begin(Macro.ATTACK, { currentTick: 0, opponentEntity: opponent });
  exec.end();
  for (const m of [Macro.APPROACH, Macro.RETREAT, Macro.STRAFE_L, Macro.STRAFE_R, Macro.JUMP]) {
    exec.begin(m, { currentTick: 100 });
    exec.end();
  }

  assert.equal(bot.pvpAttackCalls.length, 0, 'bot.pvp.attack was never called (forbidden)');
  assert.equal(bot.pathfinderGoals.length, 0, 'no pathfinder goal was set (forbidden)');
  // And the raw swing path WAS used.
  assert.equal(bot.attackCalls.length, 1, 'raw bot.attack is the only attack path');
});

test('MacroExecutor TURN_TO_LAST_SEEN calls bot.lookAt toward the stored position', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);
  const lastSeen = { x: 3.5, y: 64.0, z: 1.5 };

  const r = exec.begin(Macro.TURN_TO_LAST_SEEN, { currentTick: 0, lastSeenPosition: lastSeen });
  assert.equal(r.looked, true, 'a look happened');
  assert.equal(bot.lookAtCalls.length, 1, 'exactly one lookAt');
  assert.deepEqual(bot.lookAtCalls[0].pos, lastSeen, 'looked at the stored last-seen position');
  assert.equal(bot.lookAtCalls[0].force, true, 'force=true bypasses interpolation');
  assert.equal(bot.attackCalls.length, 0, 'TURN does not attack');
});

test('MacroExecutor TURN_TO_LAST_SEEN with no memory is a no-op (no look, no throw)', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);
  const r = exec.begin(Macro.TURN_TO_LAST_SEEN, { currentTick: 0, lastSeenPosition: null });
  assert.equal(r.looked, false, 'nothing to face -> no look');
  assert.equal(bot.lookAtCalls.length, 0);
});

test('MacroExecutor clearAll() blanket-clears control states (reset hygiene)', () => {
  const bot = makeRecordingBot();
  const exec = new MacroExecutor(bot);
  exec.begin(Macro.APPROACH, { currentTick: 0 });
  exec.clearAll();
  assert.equal(bot.clearAllCount, 1, 'used bot.clearControlStates when available');
  assert.equal(bot.controlStates.forward, false);
});

// ===========================================================================
// State assembly (bot.js) — schema-valid `state` from a drained window.
// ===========================================================================

test('buildEventsBlock normalizes a drained aggregator into a schema-valid events block', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(6);
  agg.recordDamageTaken(2);
  agg.recordOpponentDied();
  const events = buildEventsBlock(agg.drain());
  assert.deepEqual(events, {
    damage_dealt: 6,
    damage_taken: 2,
    i_died: false,
    opponent_died: true,
  });
});

test('buildEventsBlock clamps negatives and coerces missing fields (defensive)', () => {
  const events = buildEventsBlock({ damage_dealt: -5, damage_taken: NaN });
  assert.equal(events.damage_dealt, 0, 'negative clamped to 0');
  assert.equal(events.damage_taken, 0, 'NaN coerced to 0');
  assert.equal(events.i_died, false);
  assert.equal(events.opponent_died, false);
  assert.deepEqual(buildEventsBlock(null), {
    damage_dealt: 0,
    damage_taken: 0,
    i_died: false,
    opponent_died: false,
  });
});

test('assembleStateMsg builds a schema-valid `state` (validateOutbound accepts it)', () => {
  const agg = new EventAggregator();
  agg.recordDamageDealt(4);
  const msg = assembleStateMsg({
    self: {
      pos: { x: 0.5, y: 64.0, z: 0.5 },
      yaw: 0.1,
      pitch: -0.2,
      velocity: { x: 0, y: 0, z: 0 },
      on_ground: true,
      health: 18.0,
      held_item: 'iron_sword',
      attack_cooldown: 0.75,
    },
    opponent: {
      pos: { x: 3.5, y: 64.0, z: 1.5 },
      yaw: 3.14,
      pitch: 0.0,
      velocity: { x: 0, y: 0, z: 0 },
      health: 12.0,
    },
    events: agg.drain(),
    wallDistances: [8, 8, 8, 8],
    tick: 16,
    codeVersion: 'test-stamp',
  });

  // Must pass the SAME validator the transport applies before sending.
  assert.doesNotThrow(() => validateOutbound(msg));
  assert.equal(msg.type, 'state');
  assert.deepEqual(msg.self.pos, [0.5, 64.0, 0.5], 'vec3 from {x,y,z}');
  assert.equal(msg.events.damage_dealt, 4, 'carries the drained window damage');
  assert.equal(msg.opponent.health, 12.0, 'privileged raw opponent health is on the wire');
  assert.equal(msg.tick, 16);
});

test('assembleStateMsg tolerates an empty/not-ready snapshot and still validates', () => {
  const msg = assembleStateMsg({
    self: {},
    opponent: {},
    events: new EventAggregator().drain(),
    wallDistances: [],
    tick: 0,
    codeVersion: 'x',
  });
  assert.doesNotThrow(() => validateOutbound(msg));
  assert.deepEqual(msg.self.pos, [0, 0, 0]);
  assert.equal(msg.self.held_item, '');
});

// ===========================================================================
// ArenaBots.handleStep — full step wiring, no live server.
//
// Drives handleStep with mock bots and an injected tick wait. During the
// injected wait we simulate the opponent taking a hit (the wired entityHurt
// handler folds it into the aggregator), then assert the emitted `state` carries
// exactly that damage and that the NEXT step starts clean (exactly-once).
// ===========================================================================

/** A mock learner/dummy bot exposing the fields handleStep snapshots. */
function stepBot({ username, health, pos }) {
  const handlers = {};
  return {
    username,
    health,
    heldItem: { name: 'iron_sword' },
    entity: {
      username,
      position: pos,
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
      pitch: 0,
      onGround: true,
      health,
    },
    on(event, fn) {
      (handlers[event] || (handlers[event] = [])).push(fn);
    },
    off(event, fn) {
      if (!handlers[event]) return;
      handlers[event] = handlers[event].filter((h) => h !== fn);
    },
    once() {},
    emit(event, arg) {
      for (const fn of handlers[event] || []) {
        fn(arg);
      }
    },
    setControlState() {},
    clearControlStates() {},
    attack() {},
    lookAt() {},
    chat() {},
  };
}

/** A capturing transport stub: records sent messages, swallows the rest. */
function captureTransport() {
  const sent = [];
  return {
    sent,
    send(msg) {
      sent.push(msg);
    },
    on() {},
    emit(event, err) {
      if (event === 'error') {
        throw err;
      }
    },
    close() {
      return Promise.resolve();
    },
  };
}

test('handleStep runs a macro, drains the window once, and sends one schema-valid state', async () => {
  const transport = captureTransport();
  const learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0.5, y: 64, z: 0.5 } });
  const dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3.5, y: 64, z: 1.5 } });

  const arena = new ArenaBots({}, { transport });
  arena.learner = learner;
  arena.dummy = dummy;
  arena.executor = new MacroExecutor(learner, { weaponAttackSpeedTicks: 10 });
  arena.wireDamageEvents();

  // Inject a deterministic tick wait that simulates the opponent taking 5 damage
  // mid-window (its health drops from 20 to 15 and entityHurt fires once).
  arena._waitTicksImpl = async () => {
    dummy.health = 15;
    dummy.entity.health = 15;
    learner.emit('entityHurt', dummy.entity);
  };

  await arena.handleStep({ type: 'step', action: Macro.ATTACK });

  assert.equal(transport.sent.length, 1, 'exactly one state message per step');
  const msg = transport.sent[0];
  assert.doesNotThrow(() => validateOutbound(msg), 'emitted state is schema-valid');
  assert.equal(msg.events.damage_dealt, 5, 'the window owns exactly the 5 damage that landed');
  assert.equal(msg.events.damage_taken, 0);
  assert.equal(msg.tick, ACTION_REPEAT, 'tick advanced to the window boundary');

  // A second step with NO new damage must report 0 — the prior hit was already
  // drained and does not double-count.
  arena._waitTicksImpl = async () => {};
  await arena.handleStep({ type: 'step', action: Macro.IDLE });
  assert.equal(transport.sent.length, 2);
  assert.equal(transport.sent[1].events.damage_dealt, 0, 'no double-count across steps');
  assert.equal(transport.sent[1].tick, 2 * ACTION_REPEAT);
});

test('handleStep counts learner damage_taken and latches i_died on a lethal window', async () => {
  const transport = captureTransport();
  const learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0.5, y: 64, z: 0.5 } });
  const dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3.5, y: 64, z: 1.5 } });

  const arena = new ArenaBots({}, { transport });
  arena.learner = learner;
  arena.dummy = dummy;
  arena.executor = new MacroExecutor(learner, { weaponAttackSpeedTicks: 10 });
  arena.wireDamageEvents();

  arena._waitTicksImpl = async () => {
    // Learner takes a lethal hit: health 20 -> 0, the `health` event fires.
    learner.health = 0;
    learner.emit('health');
  };

  await arena.handleStep({ type: 'step', action: Macro.RETREAT });
  const msg = transport.sent[0];
  assert.equal(msg.events.damage_taken, 20, 'full health drop counted as damage_taken');
  assert.equal(msg.events.i_died, true, 'i_died latched on reaching 0 health');
  assert.doesNotThrow(() => validateOutbound(msg));
});

test('handleStep emits an error on an out-of-range action and sends no state', async () => {
  const sentErrors = [];
  const transport = {
    sent: [],
    send(msg) {
      this.sent.push(msg);
    },
    on() {},
    emit(event, err) {
      if (event === 'error') {
        sentErrors.push(err);
      }
    },
    close() {
      return Promise.resolve();
    },
  };
  const arena = new ArenaBots({}, { transport });
  arena.learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0, y: 64, z: 0 } });
  arena.dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3, y: 64, z: 0 } });
  arena.executor = new MacroExecutor(arena.learner);

  await arena.handleStep({ type: 'step', action: 99 });
  assert.equal(transport.sent.length, 0, 'no state sent for a bad action');
  assert.equal(sentErrors.length, 1, 'the out-of-range action was surfaced loudly');
});

// ===========================================================================
// W1a — wireDamageEvents() idempotency: a second call must not double-register
// handlers. Calling wireDamageEvents() twice then emitting ONE hit must count
// that hit exactly ONCE in the aggregator, not twice.
// ===========================================================================

test('W1a wireDamageEvents() is idempotent: a second call + one hit counts once (not twice)', () => {
  const learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0, y: 64, z: 0 } });
  const dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3, y: 64, z: 0 } });
  const transport = captureTransport();

  const arena = new ArenaBots({}, { transport });
  arena.learner = learner;
  arena.dummy = dummy;

  // Wire twice — simulates a reconnect / re-bind scenario.
  arena.wireDamageEvents();
  arena.wireDamageEvents();

  // Emit ONE entityHurt event representing the opponent taking 5 damage.
  dummy.health = 15;
  dummy.entity.health = 15;
  learner.emit('entityHurt', dummy.entity);

  // The aggregator must record the hit exactly once.
  const w = arena.events.drain();
  assert.equal(
    w.damage_dealt,
    5,
    'one hit after two wireDamageEvents() calls must be counted exactly once, not twice',
  );
});

// ===========================================================================
// W1b — _prevOpponentHealth re-seed on respawn: after the opponent dies and
// respawns to full health, the next genuine hit must be counted correctly
// (not dropped or under-counted due to a stale baseline).
//
// Scenario: opponent at 20hp → dies (health→0) → respawns (health→20) →
// takes a 5-damage hit (health→15). damage_dealt must equal 5.
//
// The symmetric scenario for the learner (self death → respawn → take a hit)
// is also verified.
// ===========================================================================

test('W1b opponent death→respawn→hit: post-respawn hit is counted correctly (not dropped)', () => {
  const learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0, y: 64, z: 0 } });
  const dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3, y: 64, z: 0 } });
  const transport = captureTransport();

  const arena = new ArenaBots({}, { transport });
  arena.learner = learner;
  arena.dummy = dummy;
  arena.wireDamageEvents();

  // Phase 1: opponent takes a lethal hit (20 → 0).
  dummy.health = 0;
  dummy.entity.health = 0;
  learner.emit('entityHurt', dummy.entity);
  // Consume the death-window events so the next drain is a fresh window.
  arena.events.drain();

  // Phase 2: opponent respawns (health jumps from 0 → 20). The entityHurt
  // event fires with the new full health — this is the respawn signal.
  dummy.health = 20;
  dummy.entity.health = 20;
  learner.emit('entityHurt', dummy.entity);
  // The respawn health-increase must re-seed _prevOpponentHealth; no damage
  // should be recorded for it.
  const respawnWindow = arena.events.drain();
  assert.equal(respawnWindow.damage_dealt, 0, 'a health increase (respawn) records no damage');

  // Phase 3: a genuine 5-damage hit on the freshly respawned opponent (20 → 15).
  dummy.health = 15;
  dummy.entity.health = 15;
  learner.emit('entityHurt', dummy.entity);
  const hitWindow = arena.events.drain();
  assert.equal(
    hitWindow.damage_dealt,
    5,
    'post-respawn hit is counted correctly from the re-seeded baseline (not dropped)',
  );
});

test('W1b self death→respawn→hit: post-respawn damage_taken is counted correctly', () => {
  const learner = stepBot({ username: 'learner_bot', health: 20, pos: { x: 0, y: 64, z: 0 } });
  const dummy = stepBot({ username: 'dummy_bot', health: 20, pos: { x: 3, y: 64, z: 0 } });
  const transport = captureTransport();

  const arena = new ArenaBots({}, { transport });
  arena.learner = learner;
  arena.dummy = dummy;
  arena.wireDamageEvents();

  // Phase 1: learner takes a lethal hit (20 → 0).
  learner.health = 0;
  learner.entity.health = 0;
  learner.emit('health');
  arena.events.drain();

  // Phase 2: learner respawns (health 0 → 20). The `health` event fires.
  learner.health = 20;
  learner.entity.health = 20;
  learner.emit('health');
  const respawnWindow = arena.events.drain();
  assert.equal(respawnWindow.damage_taken, 0, 'a health increase (respawn) records no damage_taken');

  // Phase 3: a genuine 3-damage hit on the freshly respawned learner (20 → 17).
  learner.health = 17;
  learner.entity.health = 17;
  learner.emit('health');
  const hitWindow = arena.events.drain();
  assert.equal(
    hitWindow.damage_taken,
    3,
    'post-respawn hit is counted correctly from the re-seeded self baseline',
  );
});

// ---------------------------------------------------------------------------
// Fire-and-forget lookAt rejection safety. bot.lookAt is async and begin() is
// sync, so a rejecting lookAt used to surface as an unhandled rejection and
// kill the whole bridge process live (TURN_TO_LAST_SEEN with a non-Vec3
// point). The executor must swallow-and-log, never propagate. This test FAILS
// THE WHOLE RUN (unhandled rejection) if the catch is missing.
// ---------------------------------------------------------------------------

test('_turnToLastSeen survives a rejecting bot.lookAt (no unhandled rejection)', async () => {
  const bot = {
    lookAt: () => Promise.reject(new TypeError('point.minus is not a function')),
  };
  const executor = new MacroExecutor(bot, { weaponAttackSpeedTicks: 12.5 });

  const looked = executor._turnToLastSeen({ x: 3.5, y: 64, z: 0.5 });
  assert.equal(looked, true, 'the turn is still reported as issued');

  // Give the rejection a tick to settle; an uncaught one aborts node --test.
  await new Promise((resolve) => setImmediate(resolve));
});

