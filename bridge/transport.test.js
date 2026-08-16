// transport.test.js — Node `node --test` suite for the T7a bridge transport +
// reset-gate logic. Runs WITHOUT a live Minecraft server.
//
// ============================================================================
// WHAT THIS FILE VERIFIES (testable now, no Paper server):
//   Framing (transport.js LineFramer):
//     - a message split BYTE-BY-BYTE across chunks reassembles exactly;
//     - several whole messages in one chunk all emit, in order;
//     - a message split across two arbitrary chunks reassembles;
//     - blank keep-alive lines are skipped, not parsed;
//     - multi-byte UTF-8 split across a chunk boundary decodes correctly;
//     - a malformed JSON line throws loudly.
//   Outbound encoding (transport.js encodeMessage / validateOutbound):
//     - state + reset_ack round-trip: encode -> frame -> deep-equal original;
//     - a missing/extra/wrong-typed outbound field throws (loud);
//     - the OPTIONAL state.opp_action_executed swing report encodes as
//       true/false/null/absent, and a non-boolean is rejected.
//   Inbound validation (transport.js validateInbound):
//     - reset / step / close accepted; unknown + outbound types rejected;
//     - step.opp_action is optional and nullable, and when present gets the
//       SAME validation as action (integer, 0..7 — N_ACTIONS stays frozen).
//   Cross-form (bridge/schema.json read from disk):
//     - every field these validators accept is declared in the canonical
//       schema, which is `additionalProperties: false`.
//   Reset gate (bot.js readbackMatchesTemplate / computeAttackCooldown):
//     - a matching readback (full health, spawn pos within epsilon, template
//       inventory, no effects) is ACCEPTED;
//     - position / health / inventory / effect mismatches are REJECTED;
//     - a null (timed-out) readback is REJECTED;
//     - the runReadbackGate loop confirms a good mock bot and times out a bad one;
//     - computeAttackCooldown maps swing tick + weapon speed into [0,1].
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (NOT covered here — see
// server/compat_check.md "Live handshake"):
//   - the real Mineflayer createBot / spawn / plugin load;
//   - TC10  reset -> step -> state round-trip over a real socket with two bots;
//   - TC14  reset determinism (same seed -> identical readback).
//   Both require the pinned Paper 1.21.1 server and are a human follow-up.
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  LineFramer,
  encodeMessage,
  validateOutbound,
  validateInbound,
  WireError,
  ACTION_MIN,
  ACTION_MAX,
  DEFAULT_PORT,
  BridgeServer,
} = require('./transport');

const {
  readbackMatchesTemplate,
  computeAttackCooldown,
  runReadbackGate,
  snapshotBotState,
} = require('./bot');

// ---------------------------------------------------------------------------
// Fixtures: canonical messages matching bridge/schema.json examples.
// ---------------------------------------------------------------------------

function makeStateMsg() {
  return {
    type: 'state',
    self: {
      pos: [0.5, 64.0, 0.5],
      yaw: 0.0,
      pitch: 0.0,
      velocity: [0.0, 0.0, 0.0],
      on_ground: true,
      health: 20.0,
      held_item: 'iron_sword',
      attack_cooldown: 1.0,
    },
    opponent: {
      pos: [3.5, 64.0, 1.5],
      yaw: 3.14,
      pitch: 0.0,
      velocity: [0.0, 0.0, 0.0],
      health: 20.0,
    },
    events: { damage_dealt: 0.0, damage_taken: 0.0, i_died: false, opponent_died: false },
    arena: { wall_distances: [8.0, 8.0, 8.0, 8.0] },
    tick: 12345,
    code_version: 'abc123',
  };
}

function makeResetAckMsg() {
  return { type: 'reset_ack', ok: true, readback: { self_hp: 20.0, opp_hp: 20.0 } };
}

// ===========================================================================
// LineFramer — partial-read buffering across packet boundaries.
// ===========================================================================

test('LineFramer reassembles a message split byte-by-byte across chunks', () => {
  const framer = new LineFramer();
  const obj = { type: 'reset', episode: 0, seed: 12345 };
  const line = JSON.stringify(obj) + '\n';
  const bytes = Buffer.from(line, 'utf8');

  const collected = [];
  for (let i = 0; i < bytes.length; i += 1) {
    // Feed exactly one byte at a time — the worst-case fragmentation.
    const out = framer.push(bytes.subarray(i, i + 1));
    collected.push(...out);
  }

  assert.equal(collected.length, 1, 'exactly one message after the final newline byte');
  assert.deepEqual(collected[0], obj);
  assert.equal(framer.pendingBytes, 0, 'buffer fully drained');
});

test('LineFramer emits multiple whole messages from a single chunk, in order', () => {
  const framer = new LineFramer();
  const a = { type: 'step', action: 0 };
  const b = { type: 'step', action: 5 };
  const c = { type: 'close' };
  const chunk = JSON.stringify(a) + '\n' + JSON.stringify(b) + '\n' + JSON.stringify(c) + '\n';

  const out = framer.push(chunk);
  assert.deepEqual(out, [a, b, c]);
  assert.equal(framer.pendingBytes, 0);
});

test('LineFramer reassembles a message split across two arbitrary chunks', () => {
  const framer = new LineFramer();
  const obj = { type: 'step', action: 7 };
  const line = JSON.stringify(obj) + '\n';
  const splitAt = 5;

  const first = framer.push(line.slice(0, splitAt));
  assert.deepEqual(first, [], 'no complete message yet (no newline seen)');
  assert.ok(framer.pendingBytes > 0, 'partial line is buffered');

  const second = framer.push(line.slice(splitAt));
  assert.deepEqual(second, [obj]);
  assert.equal(framer.pendingBytes, 0);
});

test('LineFramer carries a trailing partial message into the next chunk', () => {
  const framer = new LineFramer();
  const a = { type: 'step', action: 1 };
  const b = { type: 'step', action: 2 };
  // First chunk: one whole message + the start of the next.
  const lineA = JSON.stringify(a) + '\n';
  const lineB = JSON.stringify(b) + '\n';
  const cut = 4;

  const out1 = framer.push(lineA + lineB.slice(0, cut));
  assert.deepEqual(out1, [a], 'only the first whole message emits');
  assert.ok(framer.pendingBytes > 0, 'the partial of b is retained');

  const out2 = framer.push(lineB.slice(cut));
  assert.deepEqual(out2, [b]);
});

test('LineFramer skips blank keep-alive lines (empty and whitespace-only)', () => {
  const framer = new LineFramer();
  const a = { type: 'step', action: 3 };
  // A blank line, a whitespace line, the real message, then another blank.
  const chunk = '\n' + '   \n' + JSON.stringify(a) + '\n' + '\n';

  const out = framer.push(chunk);
  assert.deepEqual(out, [a], 'only the real message emits; blanks are skipped');
});

test('LineFramer decodes a multi-byte UTF-8 char split across a chunk boundary', () => {
  const framer = new LineFramer();
  // "é" is 0xC3 0xA9 in UTF-8 — split between the two bytes.
  const obj = { type: 'state', note: 'café' };
  const bytes = Buffer.from(JSON.stringify(obj) + '\n', 'utf8');
  // Find the multi-byte sequence and split inside it.
  const splitInsideChar = bytes.indexOf(0xc3) + 1;
  assert.ok(splitInsideChar > 0, 'fixture must contain the multi-byte char');

  const out1 = framer.push(bytes.subarray(0, splitInsideChar));
  assert.deepEqual(out1, [], 'no message before the newline');
  const out2 = framer.push(bytes.subarray(splitInsideChar));
  assert.deepEqual(out2, [obj], 'split multi-byte char reassembles correctly');
});

test('LineFramer throws loudly on a malformed JSON line', () => {
  const framer = new LineFramer();
  assert.throws(
    () => framer.push('{not valid json}\n'),
    /invalid JSON line/,
  );
});

test('LineFramer throws when a complete line decodes to a non-object', () => {
  const framer = new LineFramer();
  assert.throws(() => framer.push('42\n'), /must decode to an object/);
  const framer2 = new LineFramer();
  assert.throws(() => framer2.push('[1,2,3]\n'), /must decode to an object/);
});

// ===========================================================================
// Outbound encoding — round-trip through the framer.
// ===========================================================================

test('encodeMessage(state) round-trips through the framer to the original object', () => {
  const msg = makeStateMsg();
  const line = encodeMessage(msg);

  assert.ok(line.endsWith('\n'), 'encoded line ends with exactly one newline');
  assert.equal(line.indexOf('\n'), line.length - 1, 'no embedded newline inside the frame');
  assert.equal(line.includes(' '), false, 'compact JSON: no spaces');

  const framer = new LineFramer();
  const out = framer.push(line);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], msg, 'decoded object deep-equals the original');
});

test('encodeMessage(reset_ack) round-trips through the framer', () => {
  const msg = makeResetAckMsg();
  const framer = new LineFramer();
  const out = framer.push(encodeMessage(msg));
  assert.deepEqual(out, [msg]);
});

test('validateOutbound rejects a state missing a required field', () => {
  const msg = makeStateMsg();
  delete msg.tick;
  assert.throws(() => validateOutbound(msg), WireError);
  assert.throws(() => validateOutbound(msg), /state missing required field "tick"/);
});

test('validateOutbound rejects a state with an extra field (additionalProperties:false)', () => {
  const msg = makeStateMsg();
  msg.surprise = 1;
  assert.throws(() => validateOutbound(msg), /unexpected field "surprise"/);
});

test('validateOutbound rejects a wrong-typed field and a bad vec3 length', () => {
  const badHealth = makeStateMsg();
  badHealth.self.health = 'full';
  assert.throws(() => validateOutbound(badHealth), /state.self.health must be a finite number/);

  const badVec = makeStateMsg();
  badVec.self.pos = [0.5, 64.0]; // only 2 elements
  assert.throws(() => validateOutbound(badVec), /state.self.pos must have exactly 3 elements/);

  const nanField = makeStateMsg();
  nanField.self.yaw = NaN; // JSON cannot encode NaN
  assert.throws(() => validateOutbound(nanField), /state.self.yaw must be a finite number/);
});

test('validateOutbound rejects negative damage and a non-boolean death flag', () => {
  const neg = makeStateMsg();
  neg.events.damage_dealt = -1;
  assert.throws(() => validateOutbound(neg), /damage_dealt must be >= 0/);

  const flag = makeStateMsg();
  flag.events.i_died = 'yes';
  assert.throws(() => validateOutbound(flag), /i_died must be a boolean/);
});

test('validateOutbound rejects reset_ack with a non-object readback or non-bool ok', () => {
  const badReadback = { type: 'reset_ack', ok: false, readback: null };
  assert.throws(() => validateOutbound(badReadback), /readback must be an object/);

  const badOk = { type: 'reset_ack', ok: 'true', readback: {} };
  assert.throws(() => validateOutbound(badOk), /ok must be a boolean/);
});

test('validateOutbound rejects unknown / inbound types (only Node->Python is outbound)', () => {
  assert.throws(() => validateOutbound({ type: 'reset', episode: 0, seed: 1 }), /unknown outbound message type/);
  assert.throws(() => validateOutbound({ type: 'nonsense' }), /unknown outbound message type/);
});

test('encodeMessage allows ok:false reset_ack (the read-back timeout reply)', () => {
  const msg = { type: 'reset_ack', ok: false, readback: {} };
  const out = new LineFramer().push(encodeMessage(msg));
  assert.deepEqual(out, [msg]);
});

// ===========================================================================
// state.opp_action_executed — the OPTIONAL swing report (Node -> Python).
//
// The bridge reports here whether the opponent's swing actually fired; the
// Python side shadow-tracks the opponent's attack cooldown from it (the wire
// has no opponent cooldown channel). Without the field declared in
// validateState, encodeMessage would reject it as an unexpected property and
// the report could never be sent at all.
// ===========================================================================

test('a state WITHOUT the swing report is unchanged (the M2 path)', () => {
  const msg = makeStateMsg();
  assert.equal(Object.prototype.hasOwnProperty.call(msg, 'opp_action_executed'), false);
  const out = new LineFramer().push(encodeMessage(msg));
  assert.deepEqual(out, [msg]);
});

for (const executed of [true, false]) {
  test(`encodeMessage carries opp_action_executed:${executed}`, () => {
    const msg = { ...makeStateMsg(), opp_action_executed: executed };
    const out = new LineFramer().push(encodeMessage(msg));
    assert.deepEqual(out, [msg]);
    assert.equal(out[0].opp_action_executed, executed);
  });
}

test('encodeMessage accepts an explicit null swing report ("not reported")', () => {
  const msg = { ...makeStateMsg(), opp_action_executed: null };
  assert.doesNotThrow(() => encodeMessage(msg));
});

test('validateOutbound REJECTS a non-boolean swing report', () => {
  // A number/string would read as a truthy report downstream and silently
  // mistime the shadow cooldown, so it must fail loudly here.
  for (const bad of [1, 0, 'true', [], {}]) {
    assert.throws(
      () => validateOutbound({ ...makeStateMsg(), opp_action_executed: bad }),
      WireError,
      `opp_action_executed: ${JSON.stringify(bad)} must be rejected`,
    );
  }
});

test('validateOutbound still rejects OTHER unexpected state fields', () => {
  // Declaring one optional property must not open `state` to arbitrary ones.
  assert.throws(
    () => validateOutbound({ ...makeStateMsg(), opp_action_executed: true, extra: 1 }),
    WireError,
  );
});

// ===========================================================================
// validateInbound — the Node binding of the Python -> Node command half.
//
// Pure and exported; deliberately NOT wired into BridgeServer's receive path
// (a validation-triggered destroy would change live M2 behavior).
// ===========================================================================

test('validateInbound accepts reset / step / close', () => {
  assert.doesNotThrow(() => validateInbound({ type: 'reset', episode: 0, seed: 12345 }));
  assert.doesNotThrow(() => validateInbound({ type: 'step', action: 3 }));
  assert.doesNotThrow(() => validateInbound({ type: 'close' }));
});

test('validateInbound accepts a step WITHOUT opp_action (the M2 dummy path)', () => {
  const msg = { type: 'step', action: 7 };
  assert.equal(validateInbound(msg), msg, 'returns the message unchanged');
});

for (let oppAction = ACTION_MIN; oppAction <= ACTION_MAX; oppAction += 1) {
  test(`validateInbound accepts opp_action ${oppAction}`, () => {
    assert.doesNotThrow(() => validateInbound({ type: 'step', action: 0, opp_action: oppAction }));
  });
}

test('validateInbound accepts an explicit null opp_action ("no opponent action")', () => {
  assert.doesNotThrow(() => validateInbound({ type: 'step', action: 0, opp_action: null }));
});

test('validateInbound REJECTS an out-of-range opp_action (N_ACTIONS is frozen at 8)', () => {
  for (const bad of [8, -1, 100, -100]) {
    assert.throws(
      () => validateInbound({ type: 'step', action: 0, opp_action: bad }),
      WireError,
      `opp_action ${bad} must be rejected`,
    );
  }
});

test('validateInbound REJECTS a non-integer opp_action', () => {
  // `true` would coerce to 1 under a naive numeric check.
  for (const bad of [3.5, '3', true, false, [], {}]) {
    assert.throws(
      () => validateInbound({ type: 'step', action: 0, opp_action: bad }),
      WireError,
      `opp_action ${JSON.stringify(bad)} must be rejected`,
    );
  }
});

test('validateInbound applies the same range rule to action and opp_action', () => {
  assert.throws(() => validateInbound({ type: 'step', action: 8 }), WireError);
  assert.throws(() => validateInbound({ type: 'step', action: 8, opp_action: 8 }), WireError);
});

test('validateInbound still rejects unexpected step fields and unknown types', () => {
  assert.throws(
    () => validateInbound({ type: 'step', action: 1, opp_action: 2, extra: 7 }),
    WireError,
  );
  assert.throws(() => validateInbound({ type: 'teleport' }), WireError);
  assert.throws(() => validateInbound({ type: 'state' }), WireError, 'outbound type');
  assert.throws(() => validateInbound('not an object'), WireError);
});

test('validateInbound rejects a reset missing a required field', () => {
  assert.throws(() => validateInbound({ type: 'reset', episode: 0 }), WireError);
  assert.throws(() => validateInbound({ type: 'reset', episode: -1, seed: 0 }), WireError);
});

// ===========================================================================
// Cross-form guard: schema.json is the CANONICAL contract.
//
// The contract exists in three mutually-consistent forms (schema.md prose,
// schema.json canonical, messages.py Python) and this file is the Node binding.
// A field that reaches the prose and the code but not schema.json is a contract
// the wire does not actually carry — `additionalProperties: false` means it is
// rejected exactly when something first uses it. These tests read the REAL
// file, so deleting a property there fails here by name.
// ===========================================================================

function schemaBranch(type) {
  const schema = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'schema.json'), 'utf8'),
  );
  const branch = schema.oneOf.find((b) => b.properties.type.const === type);
  assert.ok(branch, `schema.json has no "${type}" branch`);
  return branch;
}

test('schema.json declares every field the Node validators accept', () => {
  const cases = [
    ['step', { type: 'step', action: 3, opp_action: 5 }, validateInbound],
    ['state', { ...makeStateMsg(), opp_action_executed: true }, validateOutbound],
  ];
  for (const [type, message, validator] of cases) {
    validator(message); // this file accepts the fully-populated form...
    const branch = schemaBranch(type);
    const declared = new Set(Object.keys(branch.properties));
    for (const key of Object.keys(message)) {
      assert.ok(
        declared.has(key),
        `${type}.${key} is accepted here but NOT declared in bridge/schema.json; ` +
          'additionalProperties:false means the wire rejects it',
      );
    }
    assert.equal(branch.additionalProperties, false);
  }
});

test('schema.json declares opp_action as optional, nullable, and 0..7', () => {
  const step = schemaBranch('step');
  assert.deepEqual(step.properties.opp_action.type, ['integer', 'null']);
  assert.equal(step.properties.opp_action.minimum, ACTION_MIN);
  assert.equal(step.properties.opp_action.maximum, ACTION_MAX);
  assert.deepEqual(step.required, ['type', 'action'], 'opp_action must NOT be required');
});

test('schema.json declares opp_action_executed as optional and nullable', () => {
  const state = schemaBranch('state');
  assert.deepEqual(state.properties.opp_action_executed.type, ['boolean', 'null']);
  assert.equal(state.required.includes('opp_action_executed'), false);
});

// ===========================================================================
// Frozen constant sanity (mirror schema.json / messages.py).
// ===========================================================================

test('action bounds and default port mirror the frozen contract', () => {
  assert.equal(ACTION_MIN, 0);
  assert.equal(ACTION_MAX, 7);
  assert.equal(DEFAULT_PORT, 5555, 'matches env TcpBridgeClient default port');
});

// ===========================================================================
// Reset-gate predicate — accept a match, reject mismatches and a timeout.
// ===========================================================================

const TEMPLATE = Object.freeze({
  health: 20.0,
  position: { x: 0.5, y: 64.0, z: 0.5 },
  inventory: ['iron_sword'],
  requireNoEffects: true,
});

function matchingReadback() {
  return {
    health: 20.0,
    position: { x: 0.5, y: 64.0, z: 0.5 },
    inventory: ['iron_sword'],
    effects: [],
  };
}

test('readbackMatchesTemplate ACCEPTS a fully matching readback', () => {
  assert.equal(readbackMatchesTemplate(matchingReadback(), TEMPLATE), true);
});

test('readbackMatchesTemplate accepts position within epsilon (post-/tp float noise)', () => {
  const rb = matchingReadback();
  rb.position = { x: 0.5 + 0.1, y: 64.0 - 0.1, z: 0.5 + 0.05 };
  assert.equal(readbackMatchesTemplate(rb, TEMPLATE), true);
});

test('readbackMatchesTemplate accepts health within epsilon (mid-regen tick)', () => {
  const rb = matchingReadback();
  rb.health = 19.999;
  assert.equal(readbackMatchesTemplate(rb, TEMPLATE), true);
});

test('readbackMatchesTemplate REJECTS a position past epsilon', () => {
  const rb = matchingReadback();
  rb.position = { x: 0.5 + 1.0, y: 64.0, z: 0.5 }; // 1 block off — bot did not land at spawn
  assert.equal(readbackMatchesTemplate(rb, TEMPLATE), false);
});

test('readbackMatchesTemplate REJECTS partial health (not regeared / damaged)', () => {
  const rb = matchingReadback();
  rb.health = 10.0;
  assert.equal(readbackMatchesTemplate(rb, TEMPLATE), false);
});

test('readbackMatchesTemplate REJECTS a wrong / incomplete inventory', () => {
  const missing = matchingReadback();
  missing.inventory = [];
  assert.equal(readbackMatchesTemplate(missing, TEMPLATE), false);

  const wrong = matchingReadback();
  wrong.inventory = ['wooden_sword'];
  assert.equal(readbackMatchesTemplate(wrong, TEMPLATE), false);

  const leftover = matchingReadback();
  leftover.inventory = ['iron_sword', 'dirt']; // leftover item from a prior episode
  assert.equal(readbackMatchesTemplate(leftover, TEMPLATE), false);
});

test('readbackMatchesTemplate REJECTS lingering active effects (/effect clear did not land)', () => {
  const rb = matchingReadback();
  rb.effects = ['regeneration'];
  assert.equal(readbackMatchesTemplate(rb, TEMPLATE), false);
});

test('readbackMatchesTemplate REJECTS a null readback (the timeout case)', () => {
  assert.equal(readbackMatchesTemplate(null, TEMPLATE), false);
  assert.equal(readbackMatchesTemplate(undefined, TEMPLATE), false);
});

// ===========================================================================
// runReadbackGate loop — confirm a good mock bot, time out a bad one.
// Uses an injected clock + sleep so no real time passes.
// ===========================================================================

/** A minimal mock bot exposing the shape snapshotBotState reads. */
function mockBot({ health, pos, items, effects }) {
  return {
    health,
    entity: {
      position: pos,
      effects: (effects || []).reduce((acc, name) => {
        acc[name] = { id: name };
        return acc;
      }, {}),
    },
    inventory: { items: () => (items || []).map((name) => ({ name })) },
  };
}

test('snapshotBotState reads health/position/inventory/effects off a mock bot', () => {
  const bot = mockBot({
    health: 20.0,
    pos: { x: 0.5, y: 64.0, z: 0.5 },
    items: ['iron_sword'],
    effects: [],
  });
  const snap = snapshotBotState(bot);
  assert.equal(snap.health, 20.0);
  assert.deepEqual(snap.position, { x: 0.5, y: 64.0, z: 0.5 });
  assert.deepEqual(snap.inventory, ['iron_sword']);
  assert.deepEqual(snap.effects, []);
});

test('runReadbackGate confirms (ok:true) a bot already matching the template', async () => {
  const bot = mockBot({
    health: 20.0,
    pos: { x: 0.5, y: 64.0, z: 0.5 },
    items: ['iron_sword'],
    effects: [],
  });
  const result = await runReadbackGate(bot, TEMPLATE, {
    timeoutMs: 1000,
    pollIntervalMs: 10,
    now: () => 0, // frozen clock; first poll already matches
    sleep: async () => {},
  });
  assert.equal(result.ok, true);
  assert.deepEqual(result.readback.inventory, ['iron_sword']);
});

test('runReadbackGate times out (ok:false) on a bot that never matches', async () => {
  const bot = mockBot({
    health: 10.0, // never reaches full health
    pos: { x: 9.0, y: 64.0, z: 9.0 },
    items: [],
    effects: ['poison'],
  });
  // Advance the injected clock past the deadline after the first poll.
  let t = 0;
  const result = await runReadbackGate(bot, TEMPLATE, {
    timeoutMs: 100,
    pollIntervalMs: 10,
    now: () => {
      const v = t;
      t += 60; // two reads exceed timeoutMs=100
      return v;
    },
    sleep: async () => {},
  });
  assert.equal(result.ok, false);
  assert.notEqual(result.readback, null, 'last snapshot is reported for logging');
});

test('runReadbackGate confirms once a slow command finally lands', async () => {
  // Bot starts wrong, then "lands" at spawn on the third snapshot.
  let polls = 0;
  const movingBot = {
    get health() {
      return polls >= 3 ? 20.0 : 5.0;
    },
    entity: {
      get position() {
        return polls >= 3 ? { x: 0.5, y: 64.0, z: 0.5 } : { x: 9.0, y: 64.0, z: 9.0 };
      },
      effects: {},
    },
    inventory: { items: () => (polls >= 3 ? [{ name: 'iron_sword' }] : []) },
  };
  const origSnapshot = snapshotBotState;
  // Drive polls forward via the injected sleep (counts each wait as a tick).
  const result = await runReadbackGate(movingBot, TEMPLATE, {
    timeoutMs: 10000,
    pollIntervalMs: 1,
    now: () => polls * 1, // clock advances with polls, well under timeout
    sleep: async () => {
      polls += 1;
    },
  });
  void origSnapshot;
  assert.equal(result.ok, true, 'gate confirms once the delayed reset lands');
});

// ===========================================================================
// computeAttackCooldown — swing tick + weapon speed -> [0,1].
// ===========================================================================

test('computeAttackCooldown returns 1.0 when no swing has happened yet', () => {
  assert.equal(computeAttackCooldown(100, null, 12.5), 1.0);
  assert.equal(computeAttackCooldown(100, undefined, 12.5), 1.0);
});

test('computeAttackCooldown returns 0.0 immediately after a swing', () => {
  assert.equal(computeAttackCooldown(50, 50, 12.5), 0.0);
});

test('computeAttackCooldown ramps linearly to 1.0 over the weapon period', () => {
  const speed = 12.5; // ticks for a full recharge (iron sword ~1.6 atk/s @ 20 TPS)
  assert.ok(Math.abs(computeAttackCooldown(50 + 6.25, 50, speed) - 0.5) < 1e-9);
  assert.equal(computeAttackCooldown(50 + 12.5, 50, speed), 1.0, 'fully charged at the period');
  assert.equal(computeAttackCooldown(50 + 100, 50, speed), 1.0, 'clamps at 1.0 past the period');
});

test('computeAttackCooldown is defensive against a zero/negative weapon speed', () => {
  assert.equal(computeAttackCooldown(100, 50, 0), 1.0);
  assert.equal(computeAttackCooldown(100, 50, -5), 1.0);
});

// ---------------------------------------------------------------------------
// BridgeServer._onConnection adoption (mock sockets, no real TCP). The env's
// single-reconnect recovery can race the old socket's 'close' on loopback, so
// a new connection must ADOPT (drop the stale socket) — the old refuse path
// destroyed the listenerless newcomer with an error, which is process-fatal
// and killed the bridge during the first live run.
// ---------------------------------------------------------------------------

const { EventEmitter: TestEmitter } = require('node:events');

/** Just enough net.Socket surface for _onConnection: destroy(err) mirrors the
 *  real process-fatal behavior (EventEmitter throws on unlistened 'error'). */
class FakeSocket extends TestEmitter {
  constructor() {
    super();
    this.destroyed = false;
  }

  setNoDelay() {}

  destroy(err) {
    this.destroyed = true;
    if (err !== undefined) {
      this.emit('error', err); // throws if nothing listens — like a real socket
    }
    this.emit('close');
  }
}

test('a reconnect while the old socket is still registered ADOPTS the new one', () => {
  const server = new BridgeServer();
  server.on('error', () => {}); // keep server-level errors non-fatal in the test

  const stale = new FakeSocket();
  server._onConnection(stale);
  assert.equal(server.isConnected, true);

  const fresh = new FakeSocket();
  // Old behavior: this threw (unlistened 'error' on the refused socket).
  assert.doesNotThrow(() => server._onConnection(fresh));

  assert.equal(stale.destroyed, true, 'the stale socket is dropped');
  assert.equal(server.isConnected, true, 'the new socket is the active one');

  // The adopted socket actually serves the stream end to end.
  const seen = [];
  server.on('message', (m) => seen.push(m));
  fresh.emit('data', Buffer.from('{"type":"step","action":3}\n'));
  assert.deepEqual(seen, [{ type: 'step', action: 3 }]);
});

test('adopting detaches the stale DATA feed but still signals disconnect', () => {
  const server = new BridgeServer();
  server.on('error', () => {});
  let disconnects = 0;
  server.on('disconnect', () => { disconnects += 1; });

  const stale = new FakeSocket();
  server._onConnection(stale);
  const fresh = new FakeSocket();
  server._onConnection(fresh); // adopt the newcomer, drop the stale socket

  assert.equal(disconnects, 1, 'adopt still emits the disconnect visibility signal');

  const seen = [];
  server.on('message', (m) => seen.push(m));
  // A chunk already queued on the stale socket must be IGNORED, not pushed into
  // the single shared _framer we reset for the adopted socket — otherwise these
  // stale bytes prepend into the framer and mis-frame the fresh stream. Without
  // detaching 'data', this stale line would emit a (bogus) message too.
  stale.emit('data', Buffer.from('{"type":"step","action":1}\n'));
  fresh.emit('data', Buffer.from('{"type":"step","action":3}\n'));
  assert.deepEqual(seen, [{ type: 'step', action: 3 }], 'only the fresh socket is framed');
});

test('a reconnect after the old socket closed normally still connects', () => {
  const server = new BridgeServer();
  const first = new FakeSocket();
  server._onConnection(first);
  first.destroy(); // normal disconnect: 'close' clears the registration
  assert.equal(server.isConnected, false);

  const second = new FakeSocket();
  server._onConnection(second);
  assert.equal(server.isConnected, true);
});

test('dropConnection closes the client but keeps accepting new connections', () => {
  const server = new BridgeServer();
  const first = new FakeSocket();
  server._onConnection(first);
  assert.equal(server.isConnected, true);

  server.dropConnection();
  assert.equal(first.destroyed, true, 'the active client socket is destroyed');
  assert.equal(server.isConnected, false);

  // dropConnection is idempotent and does not poison the next episode.
  server.dropConnection();
  const second = new FakeSocket();
  server._onConnection(second);
  assert.equal(server.isConnected, true, 'a fresh per-episode client connects fine');

  const seen = [];
  server.on('message', (m) => seen.push(m));
  second.emit('data', Buffer.from('{"type":"reset","episode":1,"seed":1}\n'));
  assert.deepEqual(seen, [{ type: 'reset', episode: 1, seed: 1 }]);
});
