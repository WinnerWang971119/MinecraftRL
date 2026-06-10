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
// COMBAT (T7b — macro exec + event aggregation):
//   ATTACK uses RAW `bot.attack(entity)` for a SINGLE swing, manually
//   COOLDOWN-GATED by the bridge (MacroExecutor tracks last-swing-tick); movement
//   uses time-bounded `bot.setControlState(...)` held for ACTION_REPEAT ticks
//   then cleared. We deliberately do NOT use `bot.pvp.attack` or pathfinder goals
//   (see agent/actions.py MACRO_SEMANTICS + bridge/actions.js). The bridge
//   computes `attack_cooldown` in [0,1] from the swing tick and the weapon's
//   attack-speed ticks. Damage/death events are aggregated over the ACTION_REPEAT
//   window by the pure EventAggregator (bridge/actions.js), counting each event
//   EXACTLY ONCE at the window boundary, and emitted in one `state` message.
//
// ============================================================================
// VERIFIED HERE (node --test, NO live server — see transport.test.js / actions.test.js):
//   - readbackMatchesTemplate(...) ACCEPTS a matching readback (health==max,
//     pos==spawn within epsilon, inventory==template, no active effects).
//   - readbackMatchesTemplate(...) REJECTS a position/health/inventory/effect
//     mismatch and a null (timed-out) readback.
//   - computeAttackCooldown(...) maps swing tick + weapon speed to [0,1].
//   - buildEventsBlock / assembleStateMsg shape a schema-valid `state` from a
//     snapshot + the EventAggregator drain (actions.test.js).
//   - The EventAggregator counts each window's damage/death exactly once at the
//     boundary (TC7, actions.test.js); the macro->control-state mapping and the
//     cooldown-gated single swing (actions.test.js).
// LIVE-ONLY (requires the Paper 1.21.1 server, per server/compat_check.md):
//   - The Mineflayer handshake itself (createBot, spawn, plugin load).
//   - TC7b  the real damage exchange (real entityHurt timing, real swing
//           cooldown moving health) over two opped bots.
//   - TC10  reset -> step -> state round-trip with real bots.
//   - TC14  reset determinism (same seed -> same readback).
//   These are the documented human follow-up in server/compat_check.md.
// ============================================================================
//
// Owner: T7a (Environment/bridge track) / T7b (Environment/bridge track)

'use strict';

const { BridgeServer } = require('./transport');
const {
  EventAggregator,
  MacroExecutor,
  ACTION_MIN,
  ACTION_MAX,
  IRON_SWORD_ATTACK_SPEED_TICKS,
} = require('./actions');

let codeVersionModule = null;

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
  // Bridge (Node<->Python) TCP bind. Previously read by ArenaBots but absent
  // here, so the transport always fell back to its own DEFAULT_HOST/PORT (T7a
  // dead-config note). Declared explicitly so the bind is honest and overridable;
  // these mirror transport.js DEFAULT_HOST / DEFAULT_PORT (env client :5555).
  bridgeHost: '127.0.0.1',
  bridgePort: 5555,
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
// PURE state-message assembly (unit-testable without a live server).
//
// Given a RAW per-bot snapshot (self/opponent kinematics + health), the drained
// EventAggregator block, the sensed arena geometry, the current tick, and the
// code_version stamp, assemble the ONE `state` message per the frozen schema
// (bridge/schema.json). Kept pure so `node --test` can assert the exact shape
// (and so transport.validateOutbound accepts it) without any Mineflayer.
//
// The events block is normalized here (drain may, defensively, hand back extra
// fields or wrong types from a buggy feed) so the schema's
// additionalProperties:false + type rules are never violated downstream.
// ---------------------------------------------------------------------------

/** A finite number or the given fallback (NaN/Infinity cannot be JSON-encoded). */
function finiteOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** A length-3 finite vector from a {x,y,z} or [x,y,z], else zeros. */
function toVec3(value) {
  if (Array.isArray(value)) {
    return [finiteOr(value[0], 0), finiteOr(value[1], 0), finiteOr(value[2], 0)];
  }
  if (value !== null && typeof value === 'object') {
    return [finiteOr(value.x, 0), finiteOr(value.y, 0), finiteOr(value.z, 0)];
  }
  return [0, 0, 0];
}

/**
 * Normalize a (possibly raw) aggregated events object into a schema-valid events
 * block: non-negative finite damages, boolean death flags, exactly the four
 * fields. Defensive against a malformed aggregator drain.
 *
 * @param {object} agg The EventAggregator.drain() output (or any shaped object).
 * @returns {{damage_dealt:number, damage_taken:number, i_died:boolean, opponent_died:boolean}}
 */
function buildEventsBlock(agg) {
  const dealt = agg ? finiteOr(agg.damage_dealt, 0) : 0;
  const taken = agg ? finiteOr(agg.damage_taken, 0) : 0;
  return {
    // Clamp negatives to 0: damage is non-negative by schema, and a negative
    // would be rejected by validateOutbound (loud) — clamp so a stray sign here
    // cannot crash the step, while the aggregator already ignores negatives.
    damage_dealt: dealt > 0 ? dealt : 0,
    damage_taken: taken > 0 ? taken : 0,
    i_died: Boolean(agg && agg.i_died),
    opponent_died: Boolean(agg && agg.opponent_died),
  };
}

/**
 * Assemble the frozen `state` message from raw parts. Pure: no Mineflayer, no
 * clock. The caller (handleStep) supplies a `self`/`opponent` snapshot already
 * read off the bots, the drained events, the arena probe, the end-of-window
 * tick, and the code_version stamp.
 *
 * @param {object} parts
 * @param {object} parts.self Raw self snapshot {pos, yaw, pitch, velocity,
 *   on_ground, health, held_item, attack_cooldown}.
 * @param {object} parts.opponent Raw opponent snapshot {pos, yaw, pitch,
 *   velocity, health} (health is PRIVILEGED — reward-only downstream).
 * @param {object} parts.events Drained EventAggregator block.
 * @param {number[]} parts.wallDistances Arena wall-distance probe (fixed order).
 * @param {number} parts.tick End-of-window server tick (>= 0 integer).
 * @param {string} parts.codeVersion The code_version stamp.
 * @returns {object} A schema-valid `state` message (validateOutbound accepts it).
 */
function assembleStateMsg(parts) {
  const self = parts.self || {};
  const opponent = parts.opponent || {};
  const wall = Array.isArray(parts.wallDistances) ? parts.wallDistances : [];
  return {
    type: 'state',
    self: {
      pos: toVec3(self.pos),
      yaw: finiteOr(self.yaw, 0),
      pitch: finiteOr(self.pitch, 0),
      velocity: toVec3(self.velocity),
      on_ground: Boolean(self.on_ground),
      health: finiteOr(self.health, 0),
      held_item: typeof self.held_item === 'string' ? self.held_item : '',
      attack_cooldown: finiteOr(self.attack_cooldown, 1.0),
    },
    opponent: {
      pos: toVec3(opponent.pos),
      yaw: finiteOr(opponent.yaw, 0),
      pitch: finiteOr(opponent.pitch, 0),
      velocity: toVec3(opponent.velocity),
      // PRIVILEGED raw true health — on the wire, reward-only downstream.
      health: finiteOr(opponent.health, 0),
    },
    events: buildEventsBlock(parts.events),
    arena: { wall_distances: wall.map((d) => finiteOr(d, 0)) },
    tick: Number.isInteger(parts.tick) && parts.tick >= 0 ? parts.tick : 0,
    code_version: typeof parts.codeVersion === 'string' ? parts.codeVersion : 'unknown',
  };
}

/** Lazily resolve code_version() from agent/contract_config (LIVE/runtime stamp). */
function resolveCodeVersion() {
  // The authoritative stamp lives in agent/contract_config.code_version() on the
  // Python side; the Node bridge has no equivalent build-stamp module, so for
  // kickoff it emits a fixed sentinel and the env LOGS (not rejects) any
  // mismatch. Kept behind a hook so a future Node stamp can drop in here.
  void codeVersionModule;
  return 'node-bridge';
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

    // Iron sword ~1.6 atk/s in 1.9+ combat -> ticks for a full swing recharge.
    // Imported from actions.js (IRON_SWORD_ATTACK_SPEED_TICKS) so the two
    // modules share a single source of truth and cannot drift (S2).
    this._weaponAttackSpeedTicks = IRON_SWORD_ATTACK_SPEED_TICKS;

    // Decision-window damage/death accumulator (drained once per step). PURE —
    // the live entityHurt/health handlers feed it; assembleStateMsg reads its
    // drain. See bridge/actions.js for the exactly-once-at-boundary guarantee.
    this.events = deps.events || new EventAggregator();

    // Macro executor: owns the manual ATTACK cooldown gate (last-swing tick) and
    // the control-state press/release. Bound to the learner once connected; in
    // tests a mock bot can be injected via deps.executor.
    this.executor = deps.executor || null;

    // Bound handler references retained so wireDamageEvents() can remove them
    // before re-adding on a reconnect/re-wire (W1a idempotency). Each property
    // is null until the first wire.
    this._boundOnSelfHealth = null;
    this._boundOnEntityHurt = null;
    this._boundOnLearnerDeath = null;
    this._boundOnDummyDeath = null;

    // The last-seen opponent world position the bridge remembers for
    // TURN_TO_LAST_SEEN. Updated from perception when the opponent is visible;
    // null until the opponent has been seen at least once this episode.
    this._lastSeenOpponentPos = null;

    /** End-of-window server tick (advances by ACTION_REPEAT each step). */
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

    // Bind the macro executor to the learner now that it exists.
    if (this.executor === null) {
      this.executor = new MacroExecutor(this.learner, {
        weaponAttackSpeedTicks: this._weaponAttackSpeedTicks,
      });
    }
    this.wireDamageEvents();
  }

  /**
   * Wire Mineflayer health/damage events to the pure EventAggregator (LIVE).
   *
   * Each LIVE event is recorded EXACTLY ONCE here; the aggregator then guarantees
   * each recorded event lands in exactly one decision window (drained at the
   * boundary by handleStep). The counting logic lives entirely in the pure
   * aggregator, so this wiring is a thin, server-only adapter.
   *
   *   - learner `health`     : the bot's own health changed; the drop since the
   *                            last sample is damage_taken; health==0 => i_died.
   *   - learner `entityHurt` : if the hurt entity is the opponent (dummy), the
   *                            opponent's health drop is damage_dealt; the
   *                            opponent reaching 0 => opponent_died.
   *
   * We track the previous health of each bot so each event contributes a single
   * non-negative delta (a heal is not negative damage). Mineflayer fires the
   * bot's `health` event on every health change; `entityHurt` fires per hit.
   */
  wireDamageEvents() {
    // IDEMPOTENCY (W1a): remove any previously registered handlers before
    // adding new ones so a reconnect/re-wire does not double-register and
    // cause each live hit to be counted twice. The stored bound references
    // are exactly what was passed to .on(), so .off() can find them.
    if (this.learner && typeof this.learner.off === 'function') {
      if (this._boundOnSelfHealth !== null) {
        this.learner.off('health', this._boundOnSelfHealth);
      }
      if (this._boundOnEntityHurt !== null) {
        this.learner.off('entityHurt', this._boundOnEntityHurt);
      }
      if (this._boundOnLearnerDeath !== null) {
        this.learner.off('death', this._boundOnLearnerDeath);
      }
    }
    if (this.dummy && typeof this.dummy.off === 'function') {
      if (this._boundOnDummyDeath !== null) {
        this.dummy.off('death', this._boundOnDummyDeath);
      }
    }

    // Seed previous-health trackers from the current snapshots so the first
    // event after a reset measures a real delta, not a phantom drop from 0.
    this._prevSelfHealth =
      this.learner && typeof this.learner.health === 'number'
        ? this.learner.health
        : MAX_HEALTH;
    this._prevOpponentHealth =
      this.dummy && typeof this.dummy.health === 'number'
        ? this.dummy.health
        : MAX_HEALTH;

    // Create fresh bound references for this wire so they can be removed on
    // the next call.
    this._boundOnSelfHealth = () => this._onSelfHealth();
    this._boundOnEntityHurt = (entity) => this._onEntityHurt(entity);
    this._boundOnLearnerDeath = () => this.events.recordIDied();
    this._boundOnDummyDeath = () => this.events.recordOpponentDied();

    if (this.learner && typeof this.learner.on === 'function') {
      this.learner.on('health', this._boundOnSelfHealth);
      this.learner.on('entityHurt', this._boundOnEntityHurt);
      this.learner.on('death', this._boundOnLearnerDeath);
    }
    if (this.dummy && typeof this.dummy.on === 'function') {
      // The opponent dying is authoritative from its own death event too.
      this.dummy.on('death', this._boundOnDummyDeath);
    }
  }

  /** Learner health changed: record the drop as damage_taken; 0 => i_died. */
  _onSelfHealth() {
    const now =
      this.learner && typeof this.learner.health === 'number'
        ? this.learner.health
        : this._prevSelfHealth;
    const drop = this._prevSelfHealth - now;
    if (drop > 0) {
      // Genuine damage: record it.
      this.events.recordDamageTaken(drop);
    } else if (now > this._prevSelfHealth) {
      // Health INCREASED (respawn / heal after death). Re-seed the baseline so
      // the next genuine hit is measured from the correct post-respawn health
      // rather than from the stale post-death value (W1b).
      this._prevSelfHealth = now;
      return;
    }
    if (now <= 0) {
      this.events.recordIDied();
    }
    this._prevSelfHealth = now;
  }

  /** An entity the learner can see was hurt: if it is the opponent, count it. */
  _onEntityHurt(entity) {
    if (!this._isOpponentEntity(entity)) {
      return;
    }
    const now =
      typeof entity.health === 'number' ? entity.health : this._prevOpponentHealth;
    const drop = this._prevOpponentHealth - now;
    if (drop > 0) {
      // Genuine damage: record it.
      this.events.recordDamageDealt(drop);
    } else if (now > this._prevOpponentHealth) {
      // Opponent health INCREASED (respawn / regen after death). Re-seed the
      // baseline so the next genuine hit on the opponent is measured from the
      // correct post-respawn health and is not silently under-counted (W1b).
      this._prevOpponentHealth = now;
      return;
    }
    if (now <= 0) {
      this.events.recordOpponentDied();
    }
    this._prevOpponentHealth = now;
  }

  /** True iff `entity` is the dummy/opponent bot (matched by username). */
  _isOpponentEntity(entity) {
    return (
      entity !== null &&
      typeof entity === 'object' &&
      typeof entity.username === 'string' &&
      entity.username === this.config.dummyUsername
    );
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
        await this.handleStep(msg);
        break;
      case 'close':
        // Per-episode client teardown, NOT process shutdown: the env opens a
        // fresh connection per episode and sends `close` when that client is
        // done. Drop only the client socket; the bots stay in-game and the
        // server keeps listening (the next reset re-establishes all bot state
        // anyway). Full teardown — close() — is reserved for process exit
        // (SIGINT in run.js). Treating `close` as full teardown killed the
        // live run after its first episode.
        this.transport.dropConnection();
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
    // Health does NOT reset on /tp: damage from the previous episode (or from
    // join jank like fall damage) persists on the player, and natural regen is
    // far slower than the gate's 3 s window. Force full health with an instant
    // effect — applied within a tick and never lingering in active effects, so
    // the gate's no-effects check is unaffected. Heal BOTH bots: the learner
    // for the gate, the dummy so every episode starts from equal health.
    this._sendCommand(this.learner, `/effect give ${this.config.learnerUsername} minecraft:instant_health 1 10 true`);
    this._sendCommand(this.dummy, `/effect give ${this.config.dummyUsername} minecraft:instant_health 1 10 true`);
    this._regear(this.learner);
    this._regear(this.dummy);

    // Reset per-episode state: the swing gate (so attack_cooldown starts ready),
    // the event accumulator (discard any partial pre-reset window), the tick
    // counter, the last-seen memory, and the held control states.
    if (this.executor !== null) {
      this.executor.resetCooldown();
      this.executor.clearAll();
    }
    this.events.reset();
    this._currentTick = 0;
    this._lastSeenOpponentPos = null;
    this._prevSelfHealth = MAX_HEALTH;
    this._prevOpponentHealth = MAX_HEALTH;

    // READ-BACK GATE: poll the learner until it matches the template or times out.
    const result = await runReadbackGate(this.learner, this.resetTemplate);

    this.transport.send({
      type: 'reset_ack',
      ok: result.ok,
      readback: result.readback === null ? {} : result.readback,
    });

    // The frozen reset reply is TWO messages, not one: `state` doubles as the
    // post-reset first observation (schema.md), and the env's reset() blocks on
    // _recv_state() right after an ok:true ack — without this send the env
    // waits out its full recv timeout and tears the connection down. Only on
    // ok:true: after ok:false the env immediately retries the reset, and a
    // stray state would desync its request/reply stream.
    if (result.ok) {
      this.transport.send(
        assembleStateMsg({
          self: this._snapshotSelf(),
          opponent: this._snapshotOpponent(),
          events: this.events.drain(),
          wallDistances: this._probeWallDistances(),
          tick: this._currentTick,
          codeVersion: resolveCodeVersion(),
        }),
      );
    }
  }

  /** Current attack-cooldown for the learner's held weapon, in [0,1]. */
  attackCooldown() {
    const lastSwingTick = this.executor !== null ? this.executor.lastSwingTick : null;
    return computeAttackCooldown(this._currentTick, lastSwingTick, this._weaponAttackSpeedTicks);
  }

  /**
   * Handle a `step`: run the chosen macro for ACTION_REPEAT ticks, aggregate the
   * window's damage/death events, then assemble and send ONE `state` message.
   *
   * Flow (LIVE — the tick loop needs the real server clock; the macro mapping,
   * cooldown gate, event aggregation, and state assembly are each unit-tested in
   * isolation, see actions.test.js):
   *   1. validate the action index (0..7);
   *   2. executor.begin(macro): press control states / single gated swing / look;
   *   3. wait ACTION_REPEAT ticks while the wired handlers feed the aggregator;
   *   4. executor.end(): release the transient control states;
   *   5. drain the aggregator (exactly-once at this boundary) and snapshot both
   *      bots, then send the assembled `state`.
   *
   * @param {{type:'step', action:number}} msg
   */
  async handleStep(msg) {
    const action = msg.action;
    if (!Number.isInteger(action) || action < ACTION_MIN || action > ACTION_MAX) {
      // The Python side validates outbound, so this is a defensive guard only.
      this.transport.emit('error', new Error(`step.action out of range: ${action}`));
      return;
    }

    const windowStartTick = this._currentTick;
    const opponentEntity = this._opponentEntity();

    // 2. Begin the macro for this window (gated swing happens here, at tick 0).
    if (this.executor !== null) {
      this.executor.begin(action, {
        currentTick: windowStartTick,
        opponentEntity,
        lastSeenPosition: this._lastSeenOpponentPos,
      });
    }

    // 3. Hold for ACTION_REPEAT ticks. The wired health/entityHurt handlers fold
    //    every hit into this.events during the wait; we update the last-seen
    //    memory from perception as the opponent is observed.
    await this._waitTicks(ACTION_REPEAT);
    this._updateLastSeen();

    // 4. Release the transient control states held for the window.
    if (this.executor !== null) {
      this.executor.end();
    }

    // 5. Advance the tick to the window boundary, drain the window's events
    //    (exactly once — the aggregator clears as it reads), and emit `state`.
    this._currentTick = windowStartTick + ACTION_REPEAT;
    const events = this.events.drain();
    const stateMsg = assembleStateMsg({
      self: this._snapshotSelf(),
      opponent: this._snapshotOpponent(),
      events,
      wallDistances: this._probeWallDistances(),
      tick: this._currentTick,
      codeVersion: resolveCodeVersion(),
    });
    this.transport.send(stateMsg);
  }

  /** The opponent (dummy) Mineflayer entity, the bot.attack target, or null. */
  _opponentEntity() {
    if (this.dummy && this.dummy.entity) {
      return this.dummy.entity;
    }
    return null;
  }

  /** Update the last-seen opponent position from the dummy's current position. */
  _updateLastSeen() {
    // LIVE: a real implementation reads the PerceptionFilter's visibility before
    // updating memory; for kickoff the bridge records the dummy's current world
    // position whenever it is known, which TURN_TO_LAST_SEEN then faces.
    //
    // TODO(T12): this stores the opponent's LIVE position unconditionally
    // (perfect tracking), which is a kickoff placeholder. The real last-*seen*
    // gating — only update when the opponent is within the learner's field of
    // view and not occluded — belongs to the PerceptionFilter (T12/env). Replace
    // this unconditional write with a PerceptionFilter.isVisible() guard when T12
    // lands.
    const pos = this.dummy && this.dummy.entity ? this.dummy.entity.position : null;
    if (pos && typeof pos.x === 'number' && typeof pos.y === 'number' && typeof pos.z === 'number') {
      // SNAPSHOT as a Vec3, not a plain {x,y,z}: bot.lookAt requires a Vec3
      // (it calls point.minus(...) — a plain object made the live lookAt throw
      // and the unhandled rejection killed the bridge mid-episode). clone()
      // keeps the memory a snapshot of where the opponent WAS rather than an
      // alias of the live, moving position vector.
      this._lastSeenOpponentPos =
        typeof pos.clone === 'function' ? pos.clone() : { x: pos.x, y: pos.y, z: pos.z };
    }
  }

  /** Snapshot the learner's raw self state for the `state` message (LIVE). */
  _snapshotSelf() {
    const bot = this.learner;
    const entity = bot && bot.entity ? bot.entity : null;
    return {
      pos: entity ? entity.position : null,
      yaw: entity ? entity.yaw : 0,
      pitch: entity ? entity.pitch : 0,
      velocity: entity ? entity.velocity : null,
      on_ground: entity ? Boolean(entity.onGround) : false,
      health: bot && typeof bot.health === 'number' ? bot.health : 0,
      held_item: this._heldItemName(bot),
      attack_cooldown: this.attackCooldown(),
    };
  }

  /** Snapshot the opponent's RAW state (incl. PRIVILEGED health) for the wire. */
  _snapshotOpponent() {
    const bot = this.dummy;
    const entity = bot && bot.entity ? bot.entity : null;
    return {
      pos: entity ? entity.position : null,
      yaw: entity ? entity.yaw : 0,
      pitch: entity ? entity.pitch : 0,
      velocity: entity ? entity.velocity : null,
      // PRIVILEGED raw true health — reward-only downstream, never the obs.
      health: bot && typeof bot.health === 'number' ? bot.health : 0,
    };
  }

  /** Held-item identifier string, or "" when the hand is empty / bot not ready. */
  _heldItemName(bot) {
    if (bot && bot.heldItem && typeof bot.heldItem.name === 'string') {
      return bot.heldItem.name;
    }
    return '';
  }

  /** Arena wall-distance probe (LIVE raycast). Empty until T8 wires the geometry. */
  _probeWallDistances() {
    // LIVE-ONLY: real raycasts against the arena walls in the fixed probe order.
    // Until the arena geometry lands (T8), emit an empty array (schema-valid).
    return [];
  }

  /** Resolve a tick wait against the server clock (LIVE). Injectable for tests. */
  async _waitTicks(ticks) {
    if (typeof this._waitTicksImpl === 'function') {
      await this._waitTicksImpl(ticks);
      return;
    }
    // LIVE: await the learner's physicsTick the requested number of times. Only
    // reached against a real bot; the unit tests drive handleStep with an
    // injected _waitTicksImpl (no clock needed).
    const bot = this.learner;
    if (!bot || typeof bot.once !== 'function') {
      return;
    }
    for (let i = 0; i < ticks; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((resolve) => bot.once('physicsTick', resolve));
    }
  }

  /** Send a chat command from a bot. Opped account required. */
  _sendCommand(bot, command) {
    if (bot && typeof bot.chat === 'function') {
      bot.chat(command);
    }
  }

  /**
   * Regear a bot to the template gear via opped /clear + /give. Like the other
   * reset commands these are async and unacked; the read-back gate remains the
   * source of truth that the gear actually arrived. /clear first so leftovers
   * from the previous episode (or a picked-up drop) cannot fail the gate's
   * exact-set inventory check.
   */
  _regear(bot) {
    if (!bot || typeof bot.username !== 'string') {
      return;
    }
    this._sendCommand(bot, `/clear ${bot.username}`);
    for (const item of this.resetTemplate.inventory) {
      // Template names are mineflayer item names (e.g. "iron_sword"); the
      // command needs the namespaced id.
      this._sendCommand(bot, `/give ${bot.username} minecraft:${item} 1`);
    }
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
  buildEventsBlock,
  assembleStateMsg,
  // Live gate loop + arena owner (structure for the live handshake).
  runReadbackGate,
  ArenaBots,
};
