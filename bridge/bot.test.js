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
//     - regears BOTH bots and replies reset_ack ok:true when the learner's
//       readback matches the template INCLUDING the given gear — the exact
//       configuration that failed live before _regear was implemented.
//
// WHAT STILL NEEDS THE LIVE HANDSHAKE (server/compat_check.md):
//   The real /give landing in a real inventory within the gate's 3 s timeout.
// ============================================================================

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { ArenaBots, MAX_HEALTH } = require('./bot');
const { validateOutbound } = require('./transport');

/** A minimal chat-capturing mock bot the regear/reset path can drive. */
function mockBot(username, { inventory = [] } = {}) {
  const chatLog = [];
  return {
    username,
    chatLog,
    chat: (cmd) => chatLog.push(cmd),
    health: MAX_HEALTH,
    entity: {
      position: { x: 0.5, y: 64.0, z: 0.5 },
      effects: {},
    },
    inventory: { items: () => inventory.map((name) => ({ name })) },
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
  // The learner already reads back the template gear (live, the /give lands
  // during the gate's poll window; mocks have no async command latency).
  bots.learner = mockBot('learner_bot', { inventory: ['iron_sword'] });
  bots.dummy = mockBot('dummy_bot');

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
