// bot.js — Mineflayer bot lifecycle + the reset RPC with read-back gate (T7a).
//
// Spawns TWO opped bots on the offline-mode Paper server — a learner bot and an
// idle dummy bot (usernames from config; both MUST be opped, see server/ops.json
// / T8, or the server rejects /tp, /effect, regear) — and wires them to the
// transport (transport.js). On a `reset` it teleports both bots to fixed
// spawns, clears effects, regears, and then runs the READ-BACK GATE before
// replying with `reset_ack`.
//
// THE READ-BACK GATE (why it is required):
//   Minecraft chat commands (/tp, /effect clear, regear) are async and UNACKED:
//   the server applies them some ticks later and never tells us "done". So after
//   issuing them we POLL the bot's observed state (health / position / inventory
//   / effects) until it matches the reset template within epsilon, OR a timeout
//   elapses. We reply reset_ack{ok:true, readback} on a confirmed match and
//   reset_ack{ok:false, readback} on timeout. On ok:false the env (mc_pvp_env.py)
//   treats the episode as failed-to-start; it retries once, then raises.
//
// COMBAT (structure only here; macro exec is T7b):
//   ATTACK must use RAW `bot.attack(entity)` + movement via
//   `bot.setControlState(...)`. We deliberately do NOT use `bot.pvp.attack` or
//   pathfinder goals (see agent/actions.py MACRO_SEMANTICS). The bridge computes
//   `attack_cooldown` in [0,1] from the swing tick and the weapon's attack-speed
//   ticks; T7b fills the swing hook and event aggregation. The cooldown math and
//   the hook structure are set up below.
//
// ============================================================================
// VERIFIED HERE (node --test, NO live server — see bot.test logic / transport.test.js):
//   - readbackMatchesTemplate(...) ACCEPTS a matching readback (health==max,
//     pos==spawn within epsilon, inventory==template, no active effects).
//   - readbackMatchesTemplate(...) REJECTS a position/health/inventory/effect
//     mismatch and a null (timed-out) readback.
//   - computeAttackCooldown(...) maps swing tick + weapon speed to [0,1].
// LIVE-ONLY (requires the Paper 1.21.1 server, per server/compat_check.md):
//   - The Mineflayer handshake itself (createBot, spawn, plugin load).
//   - TC10  reset -> step -> state round-trip with real bots.
//   - TC14  reset determinism (same seed -> same readback).
//   These are the documented human follow-up in server/compat_check.md.
// ============================================================================
//
// Owner: T7a (Environment/bridge track) / T7b (Environment/bridge track)

'use strict';

const { BridgeServer } = require('./transport');

// ---------------------------------------------------------------------------
// Frozen timing constants. Mirror agent/contract_config.py so the bridge can
// never drift from the Python side. (Recorded locally; the authoritative copy
// lives in contract_config.py.)
// ---------------------------------------------------------------------------

/** Vanilla server tick rate (ticks/second). Fixed by Minecraft. */
const SERVER_TPS = 20;

/** Ticks each chosen macro is held before the next decision (frame-skip). */
const ACTION_REPEAT = 4;

/** Full player health in vanilla Minecraft (the reset health template target). */
const MAX_HEALTH = 20.0;

/**
 * Read-back gate defaults. The gate polls observed bot state until it matches
 * the reset template (within epsilon) or this timeout elapses.
 */
const DEFAULT_READBACK = Object.freeze({
  // Tolerance on each position axis (blocks). Post-/tp settling + float noise.
  posEpsilon: 0.25,
  // Tolerance on health (half-hearts are 1.0; sub-unit noise only).
  healthEpsilon: 0.01,
  // Max wall-clock to wait for the gate to confirm before replying ok:false.
  timeoutMs: 3000,
  // Poll cadence while waiting.
  pollIntervalMs: 50,
});

/** Default offline-mode connection + identity config for the two bots. */
const DEFAULT_BOT_CONFIG = Object.freeze({
  host: '127.0.0.1',
  port: 25565,
  version: '1.21.1',
  auth: 'offline',
  // Must match server/ops.json so the bots are opped (commands are otherwise
  // rejected). T8 fills the real usernames/UUIDs.
  learnerUsername: 'learner_bot',
  dummyUsername: 'dummy_bot',
});

// ---------------------------------------------------------------------------
// PURE reset-gate predicate (unit-testable without a live server).
//
// Given a read-back SNAPSHOT of observed bot state and the reset TEMPLATE the
// reset is supposed to have produced, decide whether the gate is satisfied.
// Kept pure (no Mineflayer, no clock) so `node --test` can drive it with a mock
// bot state. A null/undefined readback (the timeout case — nothing confirmed)
// is always a mismatch.
//
// readback shape (free-form by contract, but the gate reads these fields):
//   { health: number,
//     position: {x,y,z},
//     inventory: string[]   // sorted item identifiers actually present
//     effects: string[] }   // active effect identifiers (empty == cleared)
// template shape:
//   { health: number,
//     position: {x,y,z},
//     inventory: string[]   // required gear, identifiers
//     requireNoEffects: boolean }
// ---------------------------------------------------------------------------

/**
 * @param {object|null|undefined} readback Observed post-reset snapshot, or null
 *   if the gate timed out before any confirmation.
 * @param {object} template The expected reset template.
 * @param {object} [tol] Tolerances ({posEpsilon, healthEpsilon}); defaults from
 *   DEFAULT_READBACK.
 * @returns {boolean} True iff the readback matches the template within tolerance.
 */
function readbackMatchesTemplate(readback, template, tol = {}) {
  // A timed-out gate confirmed nothing.
  if (readback === null || readback === undefined) {
    return false;
  }
  const posEpsilon = tol.posEpsilon !== undefined ? tol.posEpsilon : DEFAULT_READBACK.posEpsilon;
  const healthEpsilon =
    tol.healthEpsilon !== undefined ? tol.healthEpsilon : DEFAULT_READBACK.healthEpsilon;

  // Health: full (within epsilon — server may report 19.999... mid-regen tick).
  if (typeof readback.health !== 'number' || !Number.isFinite(readback.health)) {
    return false;
  }
  if (Math.abs(readback.health - template.health) > healthEpsilon) {
    return false;
  }

  // Position: each axis within epsilon of the spawn.
  if (!positionWithin(readback.position, template.position, posEpsilon)) {
    return false;
  }

  // Inventory: the regeared gear must match the template set exactly (no missing
  // gear, no leftover items from the previous episode). Order-independent.
  if (!sameItemSet(readback.inventory, template.inventory)) {
    return false;
  }

  // Effects: a fresh episode has no active effects (/effect clear took hold).
  if (template.requireNoEffects) {
    const effects = readback.effects;
    if (!Array.isArray(effects) || effects.length !== 0) {
      return false;
    }
  }

  return true;
}

/** True iff every axis of `pos` is within `epsilon` of `target`. */
function positionWithin(pos, target, epsilon) {
  if (pos === null || typeof pos !== 'object' || target === null || typeof target !== 'object') {
    return false;
  }
  for (const axis of ['x', 'y', 'z']) {
    const a = pos[axis];
    const b = target[axis];
    if (typeof a !== 'number' || !Number.isFinite(a) || typeof b !== 'number') {
      return false;
    }
    if (Math.abs(a - b) > epsilon) {
      return false;
    }
  }
  return true;
}

/** Order-independent set equality of two arrays of item identifier strings. */
function sameItemSet(actual, expected) {
  if (!Array.isArray(actual) || !Array.isArray(expected)) {
    return false;
  }
  if (actual.length !== expected.length) {
    return false;
  }
  const counts = new Map();
  for (const item of expected) {
    counts.set(item, (counts.get(item) || 0) + 1);
  }
  for (const item of actual) {
    const remaining = counts.get(item);
    if (remaining === undefined || remaining === 0) {
      return false;
    }
    counts.set(item, remaining - 1);
  }
  // Every expected count must be fully consumed (lengths already matched, but
  // this guards duplicates).
  for (const remaining of counts.values()) {
    if (remaining !== 0) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// PURE attack-cooldown computation (unit-testable without a live server).
//
// state.self.attack_cooldown is swing progress in [0,1] (1.0 == ready, i.e. a
// full-power swing is available). It is computed from how many ticks have
// elapsed since the last swing relative to the weapon's attack-speed period:
//
//     cooldown = clamp((currentTick - lastSwingTick) / weaponAttackSpeedTicks, 0, 1)
//
// The actual lastSwingTick hook is recorded by the ATTACK macro (T7b) when it
// calls bot.attack; weaponAttackSpeedTicks comes from the held weapon's attack
// speed (e.g. an iron sword in 1.9+ combat). Here we set up the math; T7b wires
// the live tick source.
// ---------------------------------------------------------------------------

/**
 * @param {number} currentTick Current server tick (>= 0).
 * @param {number|null} lastSwingTick Tick of the last swing, or null if no swing
 *   yet this episode (treated as fully ready).
 * @param {number} weaponAttackSpeedTicks Ticks for the held weapon's cooldown to
 *   fully recharge (> 0).
 * @returns {number} Swing progress clamped to [0, 1].
 */
function computeAttackCooldown(currentTick, lastSwingTick, weaponAttackSpeedTicks) {
  if (lastSwingTick === null || lastSwingTick === undefined) {
    // No swing yet -> the weapon is fully charged.
    return 1.0;
  }
  if (!(weaponAttackSpeedTicks > 0)) {
    // Degenerate / unknown weapon speed: treat as instantly ready rather than
    // dividing by zero.
    return 1.0;
  }
  const elapsed = currentTick - lastSwingTick;
  if (elapsed <= 0) {
    return 0.0;
  }
  const progress = elapsed / weaponAttackSpeedTicks;
  if (progress >= 1.0) {
    return 1.0;
  }
  return progress;
}

// ---------------------------------------------------------------------------
// Bot state observation (LIVE — reads a Mineflayer bot). Pulled out so the gate
// loop is thin and the pure predicate above does the deciding.
// ---------------------------------------------------------------------------

/**
 * Snapshot the fields the read-back gate inspects from a (live or mock) bot.
 * Tolerant of a not-yet-populated bot (returns null fields) so a poll before
 * spawn simply fails the gate rather than throwing.
 *
 * @param {object} bot A Mineflayer bot (or a mock exposing the same shape).
 * @returns {{health:number|null, position:{x,y,z}|null, inventory:string[], effects:string[]}}
 */
function snapshotBotState(bot) {
  const position =
    bot && bot.entity && bot.entity.position
      ? { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z }
      : null;

  const inventory =
    bot && typeof bot.inventory === 'object' && bot.inventory !== null && typeof bot.inventory.items === 'function'
      ? bot.inventory
          .items()
          .map((item) => item.name)
          .filter((name) => typeof name === 'string')
      : [];

  // Mineflayer exposes effects as a map of id -> {id, amplifier, duration}.
  const effects = [];
  if (bot && bot.entity && bot.entity.effects && typeof bot.entity.effects === 'object') {
    for (const key of Object.keys(bot.entity.effects)) {
      effects.push(String(key));
    }
  }

  return {
    health: bot && typeof bot.health === 'number' ? bot.health : null,
    position,
    inventory,
    effects,
  };
}

// ---------------------------------------------------------------------------
// The read-back gate loop (LIVE — uses a clock + the bot). Polls until the pure
// predicate accepts the snapshot or the timeout elapses. Factored to take an
// async sleep + clock so it stays driveable, but it is exercised end-to-end
// only against a live server (the pure predicate carries the unit-tested logic).
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Poll the bot's state until it matches `template` (within tolerance) or the
 * timeout elapses.
 *
 * @param {object} bot Mineflayer bot.
 * @param {object} template Reset template (see readbackMatchesTemplate).
 * @param {object} [options] Overrides for DEFAULT_READBACK + an injectable
 *   `now`/`sleep` for testing.
 * @returns {Promise<{ok:boolean, readback:object|null}>}
 */
async function runReadbackGate(bot, template, options = {}) {
  const timeoutMs = options.timeoutMs !== undefined ? options.timeoutMs : DEFAULT_READBACK.timeoutMs;
  const pollIntervalMs =
    options.pollIntervalMs !== undefined ? options.pollIntervalMs : DEFAULT_READBACK.pollIntervalMs;
  const tol = {
    posEpsilon: options.posEpsilon !== undefined ? options.posEpsilon : DEFAULT_READBACK.posEpsilon,
    healthEpsilon:
      options.healthEpsilon !== undefined ? options.healthEpsilon : DEFAULT_READBACK.healthEpsilon,
  };
  const now = options.now || (() => Date.now());
  const wait = options.sleep || sleep;

  const deadline = now() + timeoutMs;
  let lastSnapshot = null;
  // Poll at least once even if timeoutMs is 0.
  do {
    lastSnapshot = snapshotBotState(bot);
    if (readbackMatchesTemplate(lastSnapshot, template, tol)) {
      return { ok: true, readback: lastSnapshot };
    }
    if (now() >= deadline) {
      break;
    }
    await wait(pollIntervalMs);
  } while (now() < deadline);

  // Timed out: report the last snapshot we saw so the env can log what failed.
  return { ok: false, readback: lastSnapshot };
}

// ---------------------------------------------------------------------------
// ArenaBots — owns the two bots, the transport, and the reset RPC (LIVE).
//
// Structured for the live handshake. The decision logic it delegates to is the
// pure functions above; the Mineflayer/socket plumbing is exercised only
// against a real server (the documented human follow-up).
// ---------------------------------------------------------------------------

class ArenaBots {
  /**
   * @param {object} [config] Bot + transport config (merged over defaults).
   * @param {object} [deps] Injectable dependencies for testing/structure:
   *   - createBot: mineflayer.createBot (default lazy-required so this module
   *     loads without mineflayer installed, e.g. in CI before `npm install`).
   *   - transport: a BridgeServer (default constructed from config host/port).
   *   - resetTemplate: the reset template (health/position/inventory) for the gate.
   */
  constructor(config = {}, deps = {}) {
    this.config = { ...DEFAULT_BOT_CONFIG, ...config };
    this._createBot = deps.createBot || null;
    this.transport =
      deps.transport ||
      new BridgeServer({ host: this.config.bridgeHost, port: this.config.bridgePort });

    /** @type {object|null} The learner Mineflayer bot. */
    this.learner = null;
    /** @type {object|null} The idle dummy Mineflayer bot. */
    this.dummy = null;

    // The reset template the read-back gate checks against. Spawn/gear are fixed
    // for kickoff; T7b/T8 may refine gear and spawn jitter from the seed.
    this.resetTemplate =
      deps.resetTemplate ||
      Object.freeze({
        health: MAX_HEALTH,
        position: { x: 0.5, y: 64.0, z: 0.5 },
        inventory: ['iron_sword'],
        requireNoEffects: true,
      });

    // ATTACK cooldown hook (T7b fills lastSwingTick on each bot.attack swing).
    this._lastSwingTick = null;
    this._weaponAttackSpeedTicks = SERVER_TPS / 1.6; // iron sword ~1.6 atk/s in 1.9+.
    this._currentTick = 0;
  }

  /**
   * Connect both opped bots to the offline-mode server and load plugins.
   * LIVE-ONLY: requires a running Paper 1.21.1 server with both accounts opped
   * (server/ops.json). Lazily requires mineflayer so this module is importable
   * (and the pure logic testable) without the dependency installed.
   */
  async connect() {
    const createBot = this._createBot || require('mineflayer').createBot;
    this.learner = createBot({
      host: this.config.host,
      port: this.config.port,
      username: this.config.learnerUsername,
      version: this.config.version,
      auth: this.config.auth,
    });
    this.dummy = createBot({
      host: this.config.host,
      port: this.config.port,
      username: this.config.dummyUsername,
      version: this.config.version,
      auth: this.config.auth,
    });
    await Promise.all([waitForSpawn(this.learner), waitForSpawn(this.dummy)]);
  }

  /** Wire transport inbound messages to the reset/step/close handlers. */
  wireTransport() {
    this.transport.on('message', (msg) => {
      // Fire-and-forget: the handler replies through the transport itself.
      this._handleMessage(msg).catch((err) => this.transport.emit('error', err));
    });
  }

  async _handleMessage(msg) {
    switch (msg.type) {
      case 'reset':
        await this.handleReset(msg);
        break;
      case 'step':
        // T7b: macro exec + event aggregation -> reply with one `state`.
        break;
      case 'close':
        await this.close();
        break;
      default:
        // The Python side never sends an unknown type (it validates outbound);
        // surface it loudly if it ever happens.
        this.transport.emit('error', new Error(`unknown inbound type "${msg.type}"`));
    }
  }

  /**
   * Handle a `reset`: teleport both bots to fixed spawns, clear effects, regear,
   * then run the read-back gate and reply with reset_ack.
   *
   * Commands are async/unacked, so the gate is REQUIRED. On timeout we reply
   * ok:false and the env retries once before raising.
   *
   * @param {{type:'reset', episode:number, seed:number}} msg
   */
  async handleReset(msg) {
    // Issue the (unacked) setup commands. Both bots must be opped (server/ops.json)
    // or the server silently rejects these — that surfaces as a read-back timeout.
    const spawn = this.resetTemplate.position;
    this._sendCommand(this.learner, `/tp ${this.config.learnerUsername} ${spawn.x} ${spawn.y} ${spawn.z}`);
    this._sendCommand(this.dummy, `/tp ${this.config.dummyUsername} ${spawn.x + 3} ${spawn.y} ${spawn.z}`);
    this._sendCommand(this.learner, `/effect clear ${this.config.learnerUsername}`);
    this._sendCommand(this.dummy, `/effect clear ${this.config.dummyUsername}`);
    this._regear(this.learner);
    this._regear(this.dummy);

    // Reset the per-episode swing hook so attack_cooldown starts ready.
    this._lastSwingTick = null;

    // READ-BACK GATE: poll the learner until it matches the template or times out.
    const result = await runReadbackGate(this.learner, this.resetTemplate);

    this.transport.send({
      type: 'reset_ack',
      ok: result.ok,
      readback: result.readback === null ? {} : result.readback,
    });
  }

  /** Current attack-cooldown for the learner's held weapon, in [0,1] (T7b hook). */
  attackCooldown() {
    return computeAttackCooldown(this._currentTick, this._lastSwingTick, this._weaponAttackSpeedTicks);
  }

  /** Send a chat command from a bot. Opped account required. */
  _sendCommand(bot, command) {
    if (bot && typeof bot.chat === 'function') {
      bot.chat(command);
    }
  }

  /** Regear a bot to the template (T7b/T8 fill the real /give or kit logic). */
  _regear(bot) {
    // Placeholder: a real regear clears the inventory and gives the template
    // gear via opped /clear + /give (or a kit plugin). Left as a hook so the
    // read-back gate is the source of truth that gear actually arrived.
    void bot;
  }

  /** Tear down both bots and the transport. */
  async close() {
    if (this.learner && typeof this.learner.quit === 'function') {
      this.learner.quit();
    }
    if (this.dummy && typeof this.dummy.quit === 'function') {
      this.dummy.quit();
    }
    await this.transport.close();
  }
}

/** Resolve once a Mineflayer bot fires its `spawn` event (LIVE). */
function waitForSpawn(bot) {
  return new Promise((resolve, reject) => {
    let settled = false;
    bot.once('spawn', () => {
      if (!settled) {
        settled = true;
        resolve();
      }
    });
    bot.once('error', (err) => {
      if (!settled) {
        settled = true;
        reject(err);
      }
    });
    bot.once('kicked', (reason) => {
      if (!settled) {
        settled = true;
        reject(new Error(`bot kicked during spawn: ${reason}`));
      }
    });
  });
}

module.exports = {
  // Constants.
  SERVER_TPS,
  ACTION_REPEAT,
  MAX_HEALTH,
  DEFAULT_READBACK,
  DEFAULT_BOT_CONFIG,
  // Pure, unit-testable logic.
  readbackMatchesTemplate,
  computeAttackCooldown,
  snapshotBotState,
  // Live gate loop + arena owner (structure for the live handshake).
  runReadbackGate,
  ArenaBots,
};
