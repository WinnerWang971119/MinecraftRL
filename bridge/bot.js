// bot.js — Mineflayer bot lifecycle + the reset RPC with read-back gate (T7a).
//
// Spawns TWO opped bots on the offline-mode Paper server — a learner bot and an
// idle dummy bot (usernames from config; both MUST be opped, see server/ops.json
// / T8, or the server rejects /tp, /effect, regear) — and wires them to the
// transport (transport.js). On a `reset` it issues ONE command — this pad's
// `/function arena:reset_pad {x,z,learner,dummy}` macro, which teleports both
// bots, regears, heals, restores food and pins their spawnpoints — and then
// runs the READ-BACK GATE before replying with `reset_ack`.
//
// PAD TOPOLOGY (T9):
//   One bridge process serves ONE pad inside ONE Paper JVM. The pad ANCHOR
//   (--pad-origin "x,z", process-local, never on the wire) is the learner spawn
//   CELL, not the floor origin: learner feet land at (anchor+0.5, 64,
//   anchor+0.5), the dummy at (anchor+3.5, 64, anchor+0.5). The anchor is
//   PARSED from argv, never derived from the pad index — padAnchor(i) is the
//   launcher's sole implementation and is deliberately not mirrored here.
//   Defaults (anchor 0,0 / learner_bot / dummy_bot) reproduce the single-arena
//   path exactly.
//
// THE BRIDGE IS THE SOLE COMMAND CHANNEL AND THE SOLE RESET AUTHORITY:
//   RCON is disabled and the launcher has no console, so `/function arena:*`
//   calls ride an opped bot's chat. It is also the sole reset authority in the
//   other direction — it issues NO reset commands of its own, because two
//   overlapping reset implementations can double-apply (a bridge-side `/effect
//   clear` landing in the same tick after the datapack's instant heal and
//   saturation would strip both, silently). The datapack APPLIES the reset
//   template; the bridge VERIFIES it with the read-back gate.
//
// THE READ-BACK GATE (why it is required):
//   Minecraft chat commands (`/function`, and the /tp, /effect clear and regear
//   the function itself runs) are async and UNACKED:
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
//   computes `attack_cooldown` in [0,1] from the LATER of the last swing and the
//   reset's regear (see attackCooldown()) against the weapon's attack-speed
//   ticks. Damage/death events are aggregated over the ACTION_REPEAT
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
//   - attack_cooldown reads 0.0 on the first observation of an episode and ramps
//     to 1.0 over the weapon period, because the reset boundary can leave the
//     server's attack-strength meter uncharged (T18 / issue #28, bot.test.js).
//   - buildEventsBlock / assembleStateMsg shape a schema-valid `state` from a
//     snapshot + the EventAggregator drain (actions.test.js).
//   - The EventAggregator counts each window's damage/death exactly once at the
//     boundary (TC7, actions.test.js); the macro->control-state mapping and the
//     cooldown-gated single swing (actions.test.js).
// LIVE-ONLY (requires the Paper 1.21.1 server, per server/compat_check.md):
//   - The Mineflayer handshake itself (createBot, spawn, plugin load).
//   - TC7b  the real damage exchange (real health-event timing on each bot's
//           own connection, real swing cooldown moving health) over two opped
//           bots.
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

// ---------------------------------------------------------------------------
// OPPONENT SOURCE (T1). The bridge has exactly two kinds of opponent:
//
//   'bot'   — this pad's dummy Mineflayer bot (the training path). It has its
//             OWN connection, so its health is readable and the damage channel
//             works.
//   'human' — a challenger who joined on their own client (the exhibition
//             path). It is a player ENTITY in the learner's view and nothing
//             more: there is no second connection, and mineflayer NEVER
//             populates `entity.health` for anyone but the connected bot, so
//             its health is simply not readable here.
//
// `healthSource` on the handle states which of those two worlds a call site is
// in, so nobody "reads" a health that silently resolves to undefined.
// ---------------------------------------------------------------------------

/** Opponent is this pad's dummy Mineflayer bot (the M2/training default). */
const OPPONENT_MODE_BOT = 'bot';

/** Opponent is a human challenger's player entity (exhibition mode). */
const OPPONENT_MODE_HUMAN = 'human';

/** Health is readable from the opponent's own Mineflayer connection. */
const OPPONENT_HEALTH_OWN_CONNECTION = 'own-connection';

/** Health is NOT readable: the opponent has no connection of its own. */
const OPPONENT_HEALTH_UNAVAILABLE = 'unavailable';

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
  // PAD TOPOLOGY (T9). Process-local: neither value is ever put on the wire.
  // padOriginX/padOriginZ are the pad ANCHOR — the learner SPAWN CELL, not the
  // floor origin (learner feet land at anchor+0.5, the dummy at anchor+3.5).
  // Flat scalars rather than a nested {x,z}: Object.freeze is shallow, and a
  // shared nested default object would be one mutation away from leaking
  // between ArenaBots instances. (0,0) is today's single arena.
  padOriginX: 0,
  padOriginZ: 0,
  // 0-based pad index. Usernames and logging ONLY — never coordinates. The
  // anchor is handed to this process on argv; padAnchor(i) lives in T10's
  // launcher and is deliberately not mirrored here.
  padIndex: 0,
  // OPPONENT SOURCE (T1). Process-local, never on the wire. 'bot' is the
  // training path and the default, so an omitted key reproduces today's
  // behavior exactly. T3 (exhibition mode) is what sets 'human' in practice.
  opponentMode: OPPONENT_MODE_BOT,
  // The challenger's username in 'human' mode. null => the first player in the
  // learner's entity view that is not one of THIS pad's own bots.
  challengerUsername: null,
});

// ---------------------------------------------------------------------------
// DATAPACK MACRO BOUNDARY (T9) — pure, exported, unit-testable.
//
// The bridge is the sole command channel to the server (RCON is disabled and
// the launcher has no console), so every `/function arena:*` call is composed
// here and chatted by an opped bot. Macro arguments are validated BEFORE they
// are formatted because macro substitution is TEXTUAL and its failure modes are
// silent:
//   - a value with an NBT type suffix (`512L`, `0b`, `512.0`, `"512"`) makes
//     `$(x).5` expand to a non-coordinate, and a parse failure inside a
//     `$`-macro function ABORTS INSTANTIATION OF THE WHOLE FUNCTION — not one
//     command in it runs and nothing appears in the server log;
//   - a NEGATIVE anchor is worse: `$(x).5` expands to `-512.5`, i.e. anchor
//     MINUS half a block, which places the bots half a block off with no error
//     at all. That silent case is why these asserts exist.
// The datapack documents these preconditions and does not enforce them at
// runtime; enforcing them is this file's job.
// ---------------------------------------------------------------------------

/** Minecraft usernames as this project uses them (offline mode, ops.json). */
const MACRO_USERNAME_RE = /^[A-Za-z0-9_]{1,16}$/;

/** Render a value inside an error message without losing its type. */
function showValue(value) {
  return typeof value === 'string' ? JSON.stringify(value) : String(value);
}

/**
 * Assert a macro coordinate argument is a non-negative plain integer.
 *
 * @param {*} value The candidate coordinate.
 * @param {string} label Human label for the error (e.g. 'pad anchor x').
 * @returns {number} The value, unchanged, once proven safe to interpolate.
 * @throws {Error} Naming the offending value.
 */
function assertMacroInt(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative plain integer, got ${showValue(value)}`);
  }
  return value;
}

/**
 * Assert a macro username argument is a plain Minecraft username. It is pasted
 * between quotes inside the macro's NBT compound, so anything carrying a quote,
 * brace, comma or space would either abort the macro or, worse, rewrite its
 * argument list.
 *
 * @param {*} value The candidate username.
 * @param {string} label Human label for the error.
 * @returns {string} The value, unchanged.
 * @throws {Error} Naming the offending value.
 */
function assertMacroUsername(value, label) {
  if (typeof value !== 'string' || !MACRO_USERNAME_RE.test(value)) {
    throw new Error(
      `${label} must be a Minecraft username matching ${MACRO_USERNAME_RE.source}, got ${showValue(value)}`,
    );
  }
  return value;
}

/**
 * The RESET CAUSALITY BEACON text, as the datapack's last line in
 * spawn_learner_pad / spawn_dummy_pad emits it.
 *
 * WHY THIS EXISTS. The read-back gate verifies TEMPLATE MATCH, not causality,
 * and after a kill cycle the natural post-respawn state IS the template state:
 * the dummy respawns at its previously-pinned spawnpoint at full health with an
 * empty inventory and no effects (death clears them), and a learner that killed
 * from its spawn without moving still reads back health 20 / anchor+0.5 /
 * ['iron_sword'] / no effects. So a `reset_pad` that ABORTS AT INSTANTIATION —
 * silent at boot, total at runtime, likeliest triggered by a Paper 1.21.2 bump
 * or a "fix" to the `generic.` attribute prefix — would let BOTH gates pass and
 * the bridge ack a reset that never happened: no saturation restore (AC18
 * drifts), no knockback re-pin (the dummy stops being stationary). Invisibly,
 * and precisely under the combat probe's stationary-learner kill cycles.
 *
 * A bare respawn cannot produce this line. The datapack addresses it to the bot
 * BY NAME and stamps it with the anchor and username, so one pad's beacon can
 * never confirm another's.
 *
 * The NONCE closes the last hole: a beacon from reset N-1 that arrives after
 * N-1 gave up would otherwise satisfy reset N's latch. It is the bridge's
 * monotonic reset epoch, forwarded through the macro and stamped here, so every
 * beacon is self-identifying and a late one is simply ignored.
 *
 * @param {'learner'|'dummy'} role Which half of the reset this beacon proves.
 * @param {{x:number, z:number}} anchor The pad anchor.
 * @param {string} username The bot the datapack addressed.
 * @param {number} nonce The per-reset nonce the macro was called with.
 * @returns {string} The exact beacon text.
 */
function formatResetConfirmation(role, anchor, username, nonce) {
  const x = assertMacroInt(anchor ? anchor.x : undefined, 'pad anchor x');
  const z = assertMacroInt(anchor ? anchor.z : undefined, 'pad anchor z');
  const stamp = assertMacroInt(nonce, 'reset nonce');
  return `[arena] reset_ok ${role} ${x} ${z} ${username} ${stamp}`;
}

/**
 * The once-per-pad-per-boot geometry command: build/repair this pad's floor,
 * sub-floor and closed bedrock ring. Idempotent by the datapack's contract.
 *
 * @param {{x:number, z:number}} anchor The pad anchor.
 * @returns {string} A chat-ready command string.
 */
function formatSetupPadCommand(anchor) {
  const x = assertMacroInt(anchor ? anchor.x : undefined, 'pad anchor x');
  const z = assertMacroInt(anchor ? anchor.z : undefined, 'pad anchor z');
  return `/function arena:setup_pad {x:${x},z:${z}}`;
}

/**
 * The per-episode reset command: the datapack's `arena:reset_pad` macro applies
 * the ENTIRE reset template for both bots on this pad (sweep, teleport, regear,
 * heal, food/saturation, knockback attributes, per-bot spawnpoint).
 *
 * At the default anchor with the default usernames this expands to exactly the
 * body of the committed `arena:reset` pad-0 wrapper, modulo the leading `/`
 * that makes it a chat command (AC11; pinned by a test).
 *
 * @param {{x:number, z:number, learner:string, dummy:string, nonce:number}} args
 * @returns {string} A chat-ready command string.
 */
function formatResetPadCommand(args) {
  const spec = args || {};
  const x = assertMacroInt(spec.x, 'pad anchor x');
  const z = assertMacroInt(spec.z, 'pad anchor z');
  const learner = assertMacroUsername(spec.learner, 'learner username');
  const dummy = assertMacroUsername(spec.dummy, 'dummy username');
  // Every macro key is REQUIRED — a macro function errors if a referenced key
  // is absent — so the nonce is validated exactly like the coordinates.
  const nonce = assertMacroInt(spec.nonce, 'reset nonce');
  return `/function arena:reset_pad {x:${x},z:${z},learner:"${learner}",dummy:"${dummy}",nonce:${nonce}}`;
}

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
//
// A SWING IS NOT THE ONLY THING THAT RE-ZEROES THE SERVER'S METER (T18, issue
// #28) — see attackCooldown(), which combines this ramp with the reset's. This
// function stays a pure one-anchor ramp and is reused for both.
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

/**
 * Minimum time the confirmation wait is always given, even when the read-back
 * gates were configured with a tiny (or zero) timeout.
 *
 * The gate budget is the right ceiling — a datapack that never speaks must fail
 * inside the SAME envelope, not double it — but it is the wrong floor: when
 * both bots already match, the gates return on their first synchronous poll
 * having consumed none of their budget and having proven nothing about THIS
 * reset, while the beacon still owes a full client->server->client round trip.
 * Five polls is enough for that round trip and is bounded.
 */
const MIN_CONFIRM_WAIT_MS = 250;

/**
 * Wait until both halves of the reset are confirmed, or the budget runs out.
 *
 * Deliberately on the REAL clock and real timers, not the gate's injectable
 * `now`/`sleep`: the thing being waited for is an inbound packet, which is
 * delivered by the event loop's macrotask queue. A loop driven by an injected
 * clock that never advances would spin on microtasks and starve the very
 * delivery it is waiting for.
 *
 * @param {() => boolean} isConfirmed Predicate over the latch.
 * @param {number} budgetMs How long to keep polling.
 * @param {number} pollIntervalMs Poll cadence.
 * @returns {Promise<boolean>} True if confirmed within the budget.
 */
async function waitForConfirmation(isConfirmed, budgetMs, pollIntervalMs) {
  const deadline = Date.now() + Math.max(0, budgetMs);
  const interval = pollIntervalMs > 0 ? pollIntervalMs : DEFAULT_READBACK.pollIntervalMs;
  for (;;) {
    if (isConfirmed()) {
      return true;
    }
    if (Date.now() >= deadline) {
      return false;
    }
    // eslint-disable-next-line no-await-in-loop
    await sleep(Math.min(interval, Math.max(1, deadline - Date.now())));
  }
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
    /**
     * @type {object|null} The idle dummy Mineflayer bot. Stays null for the
     * whole run in 'human' opponent mode — the challenger joins on their own
     * client and there is no second connection to make. Read it directly only
     * for BOT-lifecycle work (spawn, reset read-back, quit); every BEHAVIORAL
     * read goes through _opponentHandle().
     */
    this.dummy = null;

    // PAD IDENTITY (T9). Validated HERE, at construction, so a malformed anchor
    // or username fails at process start with the offending value rather than
    // at the first reset — where a bad macro argument would abort the whole
    // datapack function server-side, silently.
    this.padIndex = assertMacroInt(this.config.padIndex, 'padIndex (--pad-index)');
    this.padOrigin = Object.freeze({
      x: assertMacroInt(this.config.padOriginX, 'pad anchor x (--pad-origin)'),
      z: assertMacroInt(this.config.padOriginZ, 'pad anchor z (--pad-origin)'),
    });
    assertMacroUsername(this.config.learnerUsername, 'learnerUsername');
    assertMacroUsername(this.config.dummyUsername, 'dummyUsername');

    // OPPONENT SOURCE (T1). Validated HERE, with the rest of the identity
    // config, so a typo ('humans', 'HUMAN') fails at process start instead of
    // silently resolving every behavioral read to the dummy bot for a whole
    // exhibition — or to null for a whole training run.
    const opponentMode =
      this.config.opponentMode === undefined ? OPPONENT_MODE_BOT : this.config.opponentMode;
    if (opponentMode !== OPPONENT_MODE_BOT && opponentMode !== OPPONENT_MODE_HUMAN) {
      throw new Error(
        `opponentMode must be "${OPPONENT_MODE_BOT}" or "${OPPONENT_MODE_HUMAN}", got ` +
          `${JSON.stringify(opponentMode)}`,
      );
    }
    /** @type {'bot'|'human'} Which kind of opponent this pad is fighting. */
    this.opponentMode = opponentMode;

    // The challenger's username in 'human' mode, or null for "whoever is here
    // that is not one of our own bots". A light type check is enough: unlike
    // the bot usernames this value never reaches a datapack macro (T1 resolves
    // the challenger from the learner's own entity view). Should a later task
    // put it in a macro, it must go through assertMacroUsername first.
    const challengerUsername =
      this.config.challengerUsername === undefined || this.config.challengerUsername === null
        ? null
        : this.config.challengerUsername;
    if (
      challengerUsername !== null &&
      (typeof challengerUsername !== 'string' || challengerUsername.length === 0)
    ) {
      throw new Error(
        `challengerUsername must be a non-empty string or null, got ${JSON.stringify(
          challengerUsername,
        )}`,
      );
    }
    /** @type {string|null} Pinned challenger name, or null for "first here". */
    this.challengerUsername = challengerUsername;

    // The reset template the read-back gate checks against — the bridge's
    // independent VERIFICATION of what the datapack's arena:reset_pad macro
    // APPLIES. It must mirror server/arena/.../spawn_learner_pad.mcfunction:
    // learner feet at (anchor+0.5, 64, anchor+0.5) holding exactly one iron
    // sword, healed, with no active effects. At anchor (0,0) this is the same
    // literal template as before the pad topology existed (AC11).
    this.resetTemplate =
      deps.resetTemplate ||
      Object.freeze({
        health: MAX_HEALTH,
        position: { x: this.padOrigin.x + 0.5, y: 64.0, z: this.padOrigin.z + 0.5 },
        inventory: ['iron_sword'],
        requireNoEffects: true,
      });

    // The DUMMY's read-back template. Same footprint as the learner's, offset
    // +3 on x (spawn_dummy_pad parks it at anchor+3.5), but with an EMPTY
    // inventory: the datapack declares the dummy "a passive target, no weapon"
    // and only ever /clear-s it. The bridge used to arm the dummy with an iron
    // sword itself and then expect to read one back; with the datapack as the
    // sole reset authority nothing gives it a sword, so expecting one here
    // would hard-fail this gate on EVERY reset and burn the full 3 s timeout.
    // The datapack owns the template; this mirrors it.
    //
    // THE GATE VERIFIES LESS THAN THE DATAPACK APPLIES. Health, position,
    // inventory and active effects are checked; food, saturation and the
    // knockback/movement attributes are NOT observable through mineflayer's own
    // connection in the same way and are not checked here. A passing gate means
    // "the observable template matched", not "the full template was applied" —
    // the causality beacon covers the did-it-run half, and AC18's live 20-
    // episode stationarity check remains the real backstop for the rest.
    this.dummyResetTemplate =
      deps.dummyResetTemplate ||
      Object.freeze({
        health: this.resetTemplate.health,
        position: {
          x: this.resetTemplate.position.x + 3,
          y: this.resetTemplate.position.y,
          z: this.resetTemplate.position.z,
        },
        inventory: [],
        requireNoEffects: this.resetTemplate.requireNoEffects,
      });

    // Read-back gate overrides (timeout/poll cadence + injectable now/sleep)
    // merged over DEFAULT_READBACK inside handleReset. Injectable so the gate
    // timing is tunable and, crucially, so tests can drive the gate with a fake
    // clock instead of burning real wall-clock on a never-matching mock bot.
    this._readbackOptions = deps.readbackOptions || {};

    // Iron sword ~1.6 atk/s in 1.9+ combat -> ticks for a full swing recharge.
    // Imported from actions.js (IRON_SWORD_ATTACK_SPEED_TICKS) so the two
    // modules share a single source of truth and cannot drift (S2).
    this._weaponAttackSpeedTicks = IRON_SWORD_ATTACK_SPEED_TICKS;

    // Decision-window damage/death accumulator (drained once per step). PURE —
    // the live per-bot health handlers feed it; assembleStateMsg reads its
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
    this._boundOnOpponentHealth = null;
    this._boundOnLearnerDeath = null;
    this._boundOnDummyDeath = null;
    this._boundOnLearnerMessage = null;
    this._boundOnDummyMessage = null;

    // RESET CAUSALITY LATCH. Re-armed with a fresh nonce at the top of every
    // handleReset and set by the datapack's per-bot beacon (see
    // formatResetConfirmation). The reset may only be acked ok:true once BOTH
    // halves have latched for the CURRENT nonce — a template match alone does
    // not prove the reset ran, and a beacon stamped with an older nonce proves
    // only that an older reset ran.
    this._resetConfirm = { nonce: 0, learner: false, dummy: false };

    // While true, opponent (dummy) health events are DISCARDED and the
    // opponent-health baseline is left untouched. Set for the duration of
    // handleReset: the reset heals/teleports the dummy asynchronously, and a
    // health event generated by those commands must never be recorded as
    // combat damage. The baseline is re-seeded from the dummy read-back gate
    // before the flag clears.
    this._suppressOpponentEvents = false;

    // Monotonic reset sequence. The env's reset path is reconnect-and-retry
    // (reset is idempotent by contract), so two handleReset invocations can be
    // in flight at once: a stale one still polling its gates while the retry
    // completes and the episode begins. Only the LATEST epoch may apply the
    // post-gate side effects (events.reset(), the opponent-baseline seed, the
    // suppression-flag clear, the ack) — a stale handler reaching them would
    // wipe the live episode's first-window damage.
    this._resetEpoch = 0;

    // The last-seen opponent world position the bridge remembers for
    // TURN_TO_LAST_SEEN. Updated from perception when the opponent is visible;
    // null until the opponent has been seen at least once this episode.
    this._lastSeenOpponentPos = null;

    /** End-of-window server tick (advances by ACTION_REPEAT each step). */
    this._currentTick = 0;

    // Tick (on the same post-reset clock as _currentTick) at which the RESET's
    // regear re-zeroed the server-side attack-strength meter, or null if no
    // reset has been confirmed on this instance yet. Set to 0 by handleReset's
    // epoch-guarded post-gate block; null leaves attackCooldown() reporting
    // exactly what the swing tracker alone says, which is what unit fakes and a
    // never-reset bridge should see.
    //
    // See attackCooldown() for the mechanic this models and why the anchor is
    // tick 0 rather than a wall-clock offset from the causality beacon.
    this._meterResetTick = null;
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
    // BOT LIFECYCLE — no-op for a human opponent (T1). The dummy is a second
    // Mineflayer CONNECTION; in exhibition mode the opponent is a person on
    // their own client, so spawning it would park an extra, unwanted combatant
    // in the pad. this.dummy stays null there and every behavioral read
    // resolves through _opponentHandle() to the challenger's player entity.
    if (this._opponentIsBot()) {
      this.dummy = createBot({
        host: this.config.host,
        port: this.config.port,
        username: this.config.dummyUsername,
        version: this.config.version,
        auth: this.config.auth,
      });
    }
    // waitForSpawn(null) would throw on `.once` — only wait for bots that exist.
    const spawns = [waitForSpawn(this.learner)];
    if (this.dummy !== null) {
      spawns.push(waitForSpawn(this.dummy));
    }
    await Promise.all(spawns);

    // Bind the macro executor to the learner now that it exists.
    if (this.executor === null) {
      this.executor = new MacroExecutor(this.learner, {
        weaponAttackSpeedTicks: this._weaponAttackSpeedTicks,
      });
    }
    this.wireDamageEvents();

    // Build THIS pad before any episode can start. The bridge is the sole
    // command channel (RCON is off), so the geometry call rides the opped
    // learner's chat like every other command. Idempotent by the datapack's
    // contract, so re-running it on pad 0 — already built by arena:setup at
    // datapack load — is a no-op re-fill, and a bridge restart repairs its own
    // pad for free.
    //
    // No /spawnpoint is issued here on purpose: arena:reset_pad sets a per-bot
    // spawnpoint on every reset (spawn_learner_pad / spawn_dummy_pad both end
    // with `execute as <bot> at @s run spawnpoint @s ~ ~ ~`).
    //
    // CONSTRAINT THE LAUNCHER MUST HONOR (T10): arena:setup puts ONE world
    // spawn at 0 64 0, so at fleet boot all 2N bots join inside pad 0 and only
    // leave when their own first reset_pad runs. Environmental damage is off
    // (fallDamage/fireDamage/drowningDamage/freezeDamage), but PLAYER damage is
    // not and cannot be: pad 0's learner can swing at the idle foreign bots
    // stacked around it, registering real damage_taken on THEIR bridges. So
    // every pad must be reset before ANY pad steps an episode.
    // _scanForeignPlayers() makes a violation visible; it does not prevent one.
    this._sendCommand(this.learner, formatSetupPadCommand(this.padOrigin));
  }

  /**
   * Wire Mineflayer health/damage events to the pure EventAggregator (LIVE).
   *
   * Each LIVE event is recorded EXACTLY ONCE here; the aggregator then guarantees
   * each recorded event lands in exactly one decision window (drained at the
   * boundary by handleStep). The counting logic lives entirely in the pure
   * aggregator, so this wiring is a thin, server-only adapter.
   *
   *   - learner `health` : the learner's own health changed; the drop since
   *                        the last sample is damage_taken; health==0 => i_died.
   *   - dummy `health`   : the dummy's own health changed; the drop since the
   *                        last sample is damage_dealt; health==0 =>
   *                        opponent_died.
   *
   * BOTH channels read each bot's OWN connection (`bot.health`, fed by the
   * server's update_health packet). Mineflayer NEVER populates `health` on
   * non-self entities — prismarine-entity defines no such field — so the
   * learner's entity view of the dummy can never source damage_dealt. The old
   * `entityHurt`-based recorder read exactly that always-undefined field and
   * recorded zero forever; it is deleted (not just bypassed) so a future
   * mineflayer that populates entity.health cannot silently double-count.
   *
   * We track the previous health of each bot so each event contributes a single
   * non-negative delta (a heal is not negative damage). Mineflayer fires a
   * bot's `health` event on every own-health change.
   *
   * T1 SPLIT: the opponent's health/death channel follows the OPPONENT HANDLE
   * (it exists only for an opponent with its own connection, so nothing is
   * wired for a human), while the dummy's `message` beacon follows the BOT (it
   * is reset-causality lifecycle, and _resetWasConfirmed('dummy') waits for it
   * whenever that bot is connected, whoever the opponent is).
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
      if (this._boundOnLearnerDeath !== null) {
        this.learner.off('death', this._boundOnLearnerDeath);
      }
      if (this._boundOnLearnerMessage !== null) {
        this.learner.off('message', this._boundOnLearnerMessage);
      }
    }
    // Removal is keyed on this.dummy, NOT on _opponentBot(): the dummy is the
    // only bot these handlers can ever have been added to, and if the opponent
    // mode flipped to 'human' between two wires, a handle-keyed off() would
    // skip removal and leave the dummy still feeding damage_dealt into a human
    // match. Removing from where they were added cannot leak (T1).
    if (this.dummy && typeof this.dummy.off === 'function') {
      if (this._boundOnOpponentHealth !== null) {
        this.dummy.off('health', this._boundOnOpponentHealth);
      }
      if (this._boundOnDummyDeath !== null) {
        this.dummy.off('death', this._boundOnDummyDeath);
      }
      if (this._boundOnDummyMessage !== null) {
        this.dummy.off('message', this._boundOnDummyMessage);
      }
    }

    // Seed previous-health trackers from the current snapshots so the first
    // event after a reset measures a real delta, not a phantom drop from 0.
    // Finite-only: an unspawned bot reports undefined (and a broken feed could
    // report NaN); either would poison every subsequent delta.
    this._prevSelfHealth =
      this.learner && typeof this.learner.health === 'number' && Number.isFinite(this.learner.health)
        ? this.learner.health
        : MAX_HEALTH;
    // Routed through the handle (T1): a human challenger reports no health at
    // all, which lands on the same MAX_HEALTH fallback an unspawned bot uses.
    const opponentHealth = this._opponentHealth();
    this._prevOpponentHealth = opponentHealth !== null ? opponentHealth : MAX_HEALTH;

    // Create fresh bound references for this wire so they can be removed on
    // the next call.
    this._boundOnSelfHealth = () => this._onSelfHealth();
    this._boundOnOpponentHealth = () => this._onOpponentHealth();
    this._boundOnLearnerDeath = () => this.events.recordIDied();
    // DELIBERATE: the dummy death handler is NOT gated by
    // _suppressOpponentEvents (the flag gates only _onOpponentHealth). A
    // reset-window dummy death is discarded solely by the winning handleReset's
    // post-gate events.reset(). Do NOT "fix" this by gating the death handler —
    // that would break mid-episode death detection whenever the flag is up.
    this._boundOnDummyDeath = () => this.events.recordOpponentDied();
    // Reset causality beacons, one per bot, each addressed to that bot by name.
    this._boundOnLearnerMessage = (jsonMsg) => this._onBotMessage('learner', jsonMsg);
    this._boundOnDummyMessage = (jsonMsg) => this._onBotMessage('dummy', jsonMsg);

    if (this.learner && typeof this.learner.on === 'function') {
      this.learner.on('health', this._boundOnSelfHealth);
      this.learner.on('death', this._boundOnLearnerDeath);
      this.learner.on('message', this._boundOnLearnerMessage);
    }
    // The reset causality BEACON is lifecycle, not a behavioral opponent read:
    // it proves the datapack's reset ran for the dummy bot, and
    // _resetWasConfirmed('dummy') waits for it whenever that bot exists. So it
    // follows the BOT in every opponent mode. Routing it through the handle
    // would leave the latch unlatched in exhibition mode while the dummy is
    // still connected, and every reset would then fail confirmation (T1).
    if (this.dummy && typeof this.dummy.on === 'function') {
      this.dummy.on('message', this._boundOnDummyMessage);
    }
    // The DAMAGE channel follows the OPPONENT HANDLE. damage_dealt comes from
    // the opponent's OWN health channel (see the doc block above) and its death
    // event is authoritative for opponent_died — both exist only for an
    // opponent that has its own connection. A human challenger has none, so
    // nothing is wired here and neither signal has a source; T2 sources the
    // human's death from the scoreboard instead.
    const opponentBot = this._opponentBot();
    if (opponentBot && typeof opponentBot.on === 'function') {
      opponentBot.on('health', this._boundOnOpponentHealth);
      opponentBot.on('death', this._boundOnDummyDeath);
    }
  }

  /**
   * Learner health changed: record the drop as damage_taken; 0 => i_died.
   * Same finite-only guard as _onOpponentHealth: an undefined/NaN reading
   * records nothing and leaves the baseline untouched, so one garbage sample
   * cannot permanently poison _prevSelfHealth and kill damage_taken. The one
   * DELIBERATE asymmetry with the opponent twin is that self events are never
   * reset-suppressed — the learner read-back gate plus the post-gate
   * events.reset() in handleReset already cover the reset window.
   */
  _onSelfHealth() {
    const now =
      this.learner && typeof this.learner.health === 'number' && Number.isFinite(this.learner.health)
        ? this.learner.health
        : null;
    if (now === null) {
      // Not yet populated (bot not spawned) or garbage: record nothing, leave
      // the baseline untouched.
      return;
    }
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

  /**
   * Dummy (opponent) health changed on ITS OWN connection: record the drop as
   * damage_dealt; 0 => opponent_died. Mirrors _onSelfHealth, with one
   * DELIBERATE divergence: an undefined/non-finite reading returns early
   * without touching the baseline. _onSelfHealth's `now = prev` fallback is
   * harmless there (drop 0), but folding an unpopulated value into the
   * opponent baseline is exactly the phantom-damage bug class this handler
   * replaces — do not "clean up" the asymmetry.
   */
  _onOpponentHealth() {
    if (this._suppressOpponentEvents) {
      // Mid-reset: this event was generated by the reset's own heal/teleport
      // commands, not by combat. handleReset re-seeds the baseline from the
      // dummy read-back gate before clearing the flag.
      return;
    }
    // Routed through the handle (T1). Same finite-only predicate as before; the
    // added case is an opponent with no connection of its own (a human), which
    // reports null and therefore records nothing.
    const now = this._opponentHealth();
    if (now === null) {
      // Not yet populated (bot not spawned), garbage, or an opponent with no
      // readable health: record nothing, leave the baseline untouched.
      return;
    }
    const drop = this._prevOpponentHealth - now;
    if (drop > 0) {
      // Genuine damage: record it.
      this.events.recordDamageDealt(drop);
    } else if (now > this._prevOpponentHealth) {
      // Health INCREASED (respawn / heal). Re-seed the baseline so the next
      // genuine hit is measured from the correct post-respawn health rather
      // than from the stale post-death value (W1b).
      this._prevOpponentHealth = now;
      return;
    }
    if (now <= 0) {
      this.events.recordOpponentDied();
    }
    this._prevOpponentHealth = now;
  }

  /**
   * A chat/system message arrived on one bot's own connection. The only thing
   * read here is this pad's reset causality beacon; every other message is
   * ignored.
   *
   * @param {'learner'|'dummy'} role Which bot's connection delivered it.
   * @param {*} jsonMsg A mineflayer ChatMessage (or any object with toString).
   */
  _onBotMessage(role, jsonMsg) {
    if (jsonMsg === null || jsonMsg === undefined) {
      return;
    }
    let text;
    try {
      text = typeof jsonMsg === 'string' ? jsonMsg : String(jsonMsg);
    } catch (err) {
      // A malformed ChatMessage must never take the bridge down.
      return;
    }
    // Matched against the CURRENTLY ARMED nonce: a beacon from an earlier reset
    // that arrives late proves only that the earlier reset ran.
    if (text === this._resetConfirmationText(role)) {
      this._resetConfirm[role] = true;
    }
  }

  /**
   * The beacon text this pad expects for one bot: anchor-, name- and
   * nonce-stamped.
   *
   * @param {'learner'|'dummy'} role
   * @param {number} [nonce] Defaults to the currently armed nonce.
   */
  _resetConfirmationText(role, nonce = this._resetConfirm.nonce) {
    const username = role === 'learner' ? this.config.learnerUsername : this.config.dummyUsername;
    return formatResetConfirmation(role, this.padOrigin, username, nonce);
  }

  /**
   * Whether one half of the reset proved it actually ran.
   *
   * Real mineflayer bots are EventEmitters and always receive their beacon
   * (minecraft-protocol maps the 1.21 `system_chat` packet to `systemChat`, and
   * mineflayer's chat plugin re-emits it as `message` on that bot's own
   * connection), so the missing-`on` branch only ever applies to unit fakes
   * that model no chat channel at all — the same tolerance `_sendCommand` and
   * `_trySend` already extend to mock bots and mock transports. CAVEAT: a real
   * bot that somehow lost its emitter would therefore auto-confirm. That is a
   * deliberate trade for fake-friendliness; the bot would be unable to chat the
   * reset command in the first place, so the gates would fail instead.
   *
   * @param {'learner'|'dummy'} role
   * @returns {boolean}
   */
  _resetWasConfirmed(role) {
    // BOT LIFECYCLE, deliberately still keyed on this.dummy (T1): the beacon
    // belongs to the dummy CONNECTION, not to whoever the opponent is. With a
    // human opponent this.dummy is null, so the dummy half auto-confirms via
    // the missing-`on` branch below and the reset turns on the learner alone.
    const bot = role === 'learner' ? this.learner : this.dummy;
    if (!bot || typeof bot.on !== 'function') {
      return true;
    }
    return this._resetConfirm[role] === true;
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
        // Client teardown, NOT bridge shutdown. The training env holds ONE
        // connection for the whole run and sends `close` once, at shutdown
        // (the periodic eval BORROWS that same connection, so it never sends its
        // own close). Drop only the client socket and keep both bots in-game with
        // the server still listening, so a reconnect (the env's single-reconnect
        // recovery, or a re-launched driver) resumes without re-spawning bots.
        // Full teardown via close() is reserved for process exit (SIGINT in
        // run.js); treating `close` as full teardown would kill the bridge on a
        // transient drop. NOTE: the bridge does NOT self-exit when the connection
        // goes idle — it stays up until SIGINT.
        this.transport.dropConnection();
        break;
      default:
        // The Python side never sends an unknown type (it validates outbound);
        // surface it loudly if it ever happens.
        this.transport.emit('error', new Error(`unknown inbound type "${msg.type}"`));
    }
  }

  /**
   * Send an outbound reply, tolerating a client that vanished during a slow
   * path (e.g. the up-to-3 s read-back gate). transport.send() throws
   * synchronously when there is no active connection; in the reset/step reply
   * path that throw escapes to wireTransport's .catch and is reported as a
   * bridge 'error', dropping the reply for a connection that is already gone.
   * A client disconnect is not a bridge fault, so skip the send cleanly: the
   * next connection re-establishes all state via reset.
   *
   * @param {object} msg A schema-valid outbound message.
   * @returns {boolean} True if the message was written, false if skipped.
   */
  _trySend(msg) {
    // Only bail when the transport EXPLICITLY reports no connection; mock
    // transports in unit tests omit isConnected and must still send.
    if (this.transport.isConnected === false) {
      return false;
    }
    try {
      this.transport.send(msg);
      return true;
    } catch (err) {
      // The socket dropped between the isConnected check and the write (TOCTOU),
      // or the transport has no live connection. A disconnect is not a bridge
      // fault — do not emit 'error'; the reply for a gone client is simply lost.
      return false;
    }
  }

  /**
   * The per-episode reset command for THIS pad. Composed (and its macro
   * arguments re-validated) on every call so a later mutation of the config
   * cannot slip an unchecked value into the macro.
   *
   * @returns {string} `/function arena:reset_pad {x:..,z:..,learner:"..",dummy:".."}`
   */
  _resetPadCommand(nonce = this._resetConfirm.nonce) {
    return formatResetPadCommand({
      x: this.padOrigin.x,
      z: this.padOrigin.z,
      learner: this.config.learnerUsername,
      dummy: this.config.dummyUsername,
      nonce,
    });
  }

  /**
   * Log any player in the learner's entity view that is not one of THIS pad's
   * two bots (T12's cross-pad isolation evidence, AC13).
   *
   * `dummy.on('health')` records a health DROP with no attacker attribution, so
   * a learner that reached a neighbouring pad would silently credit its damage
   * to that pad's policy. Walls and ≥512-block spacing make that impossible by
   * construction; this scan is the observable that PROVES it, emitted on the
   * bridge's stderr (never on the frozen wire) once per reset.
   *
   * @returns {string[]} Foreign usernames seen, in first-seen order.
   */
  _scanForeignPlayers() {
    const own = new Set([this.config.learnerUsername, this.config.dummyUsername]);
    const foreign = [];
    const entities =
      this.learner && this.learner.entities && typeof this.learner.entities === 'object'
        ? this.learner.entities
        : null;
    if (entities === null) {
      return foreign;
    }
    for (const key of Object.keys(entities)) {
      const entity = entities[key];
      if (!entity || entity.type !== 'player') {
        continue;
      }
      const name = typeof entity.username === 'string' ? entity.username : null;
      if (name === null || own.has(name) || foreign.includes(name)) {
        continue;
      }
      foreign.push(name);
    }
    if (foreign.length > 0) {
      // Machine-greppable, pad-tagged: eval/benchmark.py (T12) consumes this.
      console.error(`[bridge] pad ${this.padIndex} foreign_players ${foreign.join(',')}`);
    }
    return foreign;
  }

  /**
   * Handle a `reset`: issue this pad's `arena:reset_pad` macro, then run the
   * read-back gates (learner AND dummy) and reply with reset_ack.
   *
   * Commands are async/unacked, so the gate is REQUIRED. On timeout we reply
   * ok:false and the env retries once before raising.
   *
   * @param {{type:'reset', episode:number, seed:number}} msg
   */
  async handleReset(msg) {
    // Claim a reset epoch. Reset is reconnect-and-retry on the env side, so a
    // retry can arrive while this invocation is still awaiting its gates; from
    // that moment this handler is STALE and must apply no further side effects
    // (see the check after the gates). Incremented synchronously at entry, so
    // the newest invocation always owns the highest epoch.
    const epoch = ++this._resetEpoch;

    // ONE command, ONE reset authority (T9). The datapack's arena:reset_pad
    // macro applies the whole template for BOTH bots on this pad: entity
    // sweep, teleport to the anchor, /clear + regear, effect clear, instant
    // health, food/saturation, the dummy's knockback/movement attributes, and
    // a per-bot /spawnpoint. It is issued through the opped learner's chat
    // because the bridge is the sole command channel (RCON is disabled and the
    // launcher has no console).
    //
    // The bridge deliberately issues NO reset commands of its own any more.
    // Two overlapping reset implementations was the real hazard: the bridge's
    // old unconditional `/effect clear` could land in the same tick AFTER the
    // datapack's instant_health + saturation gives and strip them before their
    // single tick applied — silently voiding the food restore that AC18's
    // cross-episode health stationarity rides on, with no error anywhere. The
    // datapack's clear-FIRST-then-give ordering is correct by construction and
    // is now the only ordering in play. What stays here is the read-back gate:
    // the datapack APPLIES the reset, the bridge VERIFIES it independently.
    //
    // Commands remain async and UNACKED, so the gate below is still required.
    //
    // Arm the causality latch BEFORE issuing the command: the gates prove the
    // observed state MATCHES the template, the beacons prove the datapack
    // actually produced it. Both are needed — see formatResetConfirmation. The
    // epoch doubles as the per-reset nonce, so a beacon can be attributed to
    // exactly one reset.
    this._resetConfirm = { nonce: epoch, learner: false, dummy: false };
    const confirmStartedAt = Date.now();
    this._sendCommand(this.learner, this._resetPadCommand(epoch));

    // Reset per-episode state: the swing gate (so no previous episode's swing
    // is still cooling), the tick counter, the last-seen memory, and the held
    // control states. (The event accumulator is reset AFTER the read-back
    // gates, so everything the reset itself generated — including events fired
    // while the gates poll — is discarded before the ack.)
    //
    // NOTE (T18): clearing the swing gate no longer means attack_cooldown starts
    // at 1.0. The regear can re-zero the SERVER's attack-strength meter, so the
    // reported value also ramps from _meterResetTick, seeded in the
    // epoch-guarded post-gate block below. See attackCooldown().
    if (this.executor !== null) {
      this.executor.resetCooldown();
      this.executor.clearAll();
    }
    this._currentTick = 0;
    this._lastSeenOpponentPos = null;
    this._prevSelfHealth = MAX_HEALTH;
    // Interim seed only: the authoritative baseline comes from the dummy
    // read-back gate below.
    this._prevOpponentHealth = MAX_HEALTH;

    // READ-BACK GATES: poll BOTH bots until each matches its template or times
    // out. The dummy gate (health + position) exists because the reset heals
    // the dummy asynchronously — acking while it is still hurt would let the
    // first real hit be measured against a phantom baseline. The dummy template
    // (this.dummyResetTemplate) mirrors what spawn_dummy_pad actually applies:
    // the learner spawn offset +3 on x, healed, effects cleared, and an EMPTY
    // inventory — the datapack gives the dummy no weapon.
    let result;
    let dummyResult;
    let confirmed = false;
    try {
      // Discard opponent health events for the whole reset window: the reset
      // commands heal/teleport the dummy asynchronously, and a health event
      // they generate must never be recorded as combat damage. Set as the
      // FIRST statement inside the try so the flag cannot be raised and then
      // stranded by a throw outside the finally's reach (e.g. bot.chat() on a
      // disconnected client in the command section above). Setting it here is
      // not late: no await occurs between handler entry and this line, so no
      // event can be delivered before it. Cleared in the finally, only after
      // the dummy read-back gate re-seeds the baseline.
      this._suppressOpponentEvents = true;

      // The dummy gate is BOT LIFECYCLE and no-ops for a human opponent (T1):
      // it verifies a Mineflayer connection's read-back, and a challenger has
      // none — runReadbackGate(null, ...) would poll the FULL timeout and then
      // fail every reset. Skipped as vacuously ok there, which falls through to
      // the readback===null branch below and leaves the opponent baseline at
      // MAX_HEALTH. Keyed on the opponent MODE and not on this.dummy being
      // null, so in 'bot' mode a missing dummy still fails the gate exactly as
      // it does today.
      [result, dummyResult] = await Promise.all([
        runReadbackGate(this.learner, this.resetTemplate, this._readbackOptions),
        this._opponentIsBot()
          ? runReadbackGate(this.dummy, this.dummyResetTemplate, this._readbackOptions)
          : Promise.resolve({ ok: true, readback: null }),
      ]);

      // STALE-EPOCH GUARD: a retry reset superseded this invocation while its
      // gates were polling. The retry (or a yet-newer one) now owns the
      // episode: it will seed the baseline, discard the reset window, clear
      // the suppression flag, and ack. Applying OUR post-gate effects here —
      // in particular events.reset() — could wipe real first-window damage
      // from the episode the retry already started. Bail without acking: an
      // ack for a superseded reset would desync the request/reply stream.
      if (epoch !== this._resetEpoch) {
        return;
      }

      // CAUSALITY WAIT. The gates can return on their FIRST SYNCHRONOUS POLL
      // when both bots already match — which is exactly the post-kill posture
      // the beacon exists to police — so at this point no inbound packet has
      // been processed at all and the beacon cannot have arrived yet. Checking
      // the latch here without waiting would fail every healthy reset. Wait for
      // it, bounded by what is LEFT of the gate budget (floored at
      // MIN_CONFIRM_WAIT_MS) so a datapack that never speaks still fails inside
      // the same envelope instead of doubling it.
      //
      // Waiting HERE, inside the try, is deliberate: suppression is still up,
      // so dummy health events generated by the reset during this window are
      // discarded, and events.reset() below still runs after the wait.
      if (result.ok && dummyResult.ok) {
        const gateTimeoutMs =
          this._readbackOptions.timeoutMs !== undefined
            ? this._readbackOptions.timeoutMs
            : DEFAULT_READBACK.timeoutMs;
        const remainingMs = gateTimeoutMs - (Date.now() - confirmStartedAt);
        confirmed = await waitForConfirmation(
          () => this._resetWasConfirmed('learner') && this._resetWasConfirmed('dummy'),
          Math.max(remainingMs, MIN_CONFIRM_WAIT_MS),
          this._readbackOptions.pollIntervalMs !== undefined
            ? this._readbackOptions.pollIntervalMs
            : DEFAULT_READBACK.pollIntervalMs,
        );
      }

      // The wait is a new await point, so re-check the epoch: a retry may have
      // superseded this handler while it was waiting, and a stale handler must
      // still apply NONE of its post-gate effects.
      if (epoch !== this._resetEpoch) {
        return;
      }

      // Seed the opponent baseline from the CONFIRMED dummy read-back, so the
      // first post-reset delta is measured against what the server actually
      // reports rather than an assumed constant.
      this._prevOpponentHealth =
        dummyResult.ok &&
        dummyResult.readback !== null &&
        typeof dummyResult.readback.health === 'number' &&
        Number.isFinite(dummyResult.readback.health)
          ? dummyResult.readback.health
          : MAX_HEALTH;

      // Anchor the learner's attack-strength meter at the start of this
      // episode's tick frame (T18, issue #28). Two things can have left the
      // SERVER's meter uncharged here — the datapack's regear, and the previous
      // episode's final swing, which executor.resetCooldown() above just made
      // the bridge forget — and the beacon that confirmed this reset is emitted
      // on the line AFTER the regear, so both are at or before this point.
      // _currentTick was set to 0 above and the first step's window begins
      // there, so tick 0 IS that moment expressed on this episode's clock;
      // attackCooldown() ramps from it over the weapon period.
      //
      // Placed inside the epoch guard with the other three post-gate effects
      // for consistency, not because a stale handler could corrupt it today: a
      // stale handler would write the same constant 0, and _currentTick is
      // owned by the newest reset. The guard is what keeps that true if this
      // anchor ever stops being a constant.
      this._meterResetTick = 0;

      // Discard every event the reset generated (teleport jank, heals, a
      // dummy respawn death) before acknowledging. No step can interleave
      // here — the transport serves one client request at a time — so no real
      // combat window is thrown away.
      this.events.reset();
    } finally {
      // Un-suppress even if a gate throws: a stuck flag would silently zero
      // damage_dealt for the rest of the run — the exact bug class this
      // handler chain exists to eliminate. EPOCH-GUARDED: a stale handler
      // must NOT clear the flag out from under a newer reset that is still
      // mid-gate (that newer handler's own finally clears it). The latest
      // epoch always reaches its own finally, so the flag can never stick.
      if (epoch === this._resetEpoch) {
        this._suppressOpponentEvents = false;
      }
    }

    // The episode may start only if BOTH bots confirmed their reset state AND
    // both halves of the datapack reset proved they ran. A not-yet-healed dummy
    // is as fatal to the damage channel as a misplaced learner — and a state
    // that merely LOOKS reset (post-kill respawn) is as fatal as either, which
    // is what the beacons rule out. On ok:false the env retries once, then
    // raises. The ack's readback stays the LEARNER's (frozen wire shape).
    //
    // `confirmed` was resolved by the bounded causality wait above, inside the
    // try. It is false whenever a gate failed (the wait is skipped then — the
    // gate failure already tells the env to retry) and whenever the datapack
    // stayed silent for the whole remaining budget.
    if (!confirmed && result.ok && dummyResult.ok) {
      // Both gates matched but the datapack never spoke: the classic silent
      // failure — a macro that aborted at instantiation while the post-kill
      // state happened to look exactly like a fresh reset. Name it loudly.
      console.error(
        `[bridge] pad ${this.padIndex} reset NOT confirmed by the datapack ` +
          `(learner=${this._resetWasConfirmed('learner')}, dummy=${this._resetWasConfirmed('dummy')}) ` +
          'though both read-back gates matched — arena:reset_pad may have aborted at instantiation',
      );
    }
    const ok = result.ok && dummyResult.ok && confirmed;
    const acked = this._trySend({
      type: 'reset_ack',
      ok,
      readback: result.readback === null ? {} : result.readback,
    });

    // Cross-pad isolation evidence (AC13/T12): once per episode, name anything
    // in view that is not one of this pad's two bots. Logged, never on the wire.
    this._scanForeignPlayers();

    // The frozen reset reply is TWO messages, not one: `state` doubles as the
    // post-reset first observation (schema.md), and the env's reset() blocks on
    // _recv_state() right after an ok:true ack — without this send the env
    // waits out its full recv timeout and tears the connection down. Only on
    // ok:true: after ok:false the env immediately retries the reset, and a
    // stray state would desync its request/reply stream. Skip it if the ack
    // didn't go out (the client disconnected during the gate): a state with no
    // matching reset_ack would desync the env's request/reply stream too.
    if (acked && ok) {
      this._trySend(
        assembleStateMsg({
          self: this._snapshotSelf(),
          opponent: this._snapshotOpponent(),
          events: this.events.drain(),
          wallDistances: this._probeWallDistances(),
          tick: this._serverTick(),
          codeVersion: resolveCodeVersion(),
        }),
      );
    }
  }

  /**
   * Current attack-cooldown for the learner's held weapon, in [0,1] (1.0 ==
   * fully charged, a full-power swing is available; 0.0 == just re-zeroed).
   *
   * TWO EVENTS RE-ZERO THE SERVER'S METER, AND THE WIRE MUST REFLECT BOTH
   * (T18, issue #28). The reported value is the ramp from whichever happened
   * LAST, so it is the MINIMUM of the two ramps:
   *
   *   1. the learner's last swing this episode (executor.lastSwingTick), and
   *   2. the RESET boundary (this._meterResetTick) — which stands for the
   *      regear AND for the previous episode's final swing, since handleReset
   *      clears lastSwingTick and the bridge forgets that one ever happened.
   *
   * THE MECHANIC, VERIFIED AT PRIMARY SOURCE — not from memory. Paper 1.21.1 is
   * Mojang-mapped, so `javap -p -c net/minecraft/world/entity/player/Player.class`
   * out of server/versions/1.21.1/paper-1.21.1.jar reads Player.tick() directly:
   *
   *     this.attackStrengthTicker++;
   *     ItemStack main = this.getMainHandItem();
   *     if (!ItemStack.matches(this.lastItemInMainHand, main)) {
   *       if (!ItemStack.isSameItem(this.lastItemInMainHand, main)) {
   *         this.resetAttackStrengthTicker();   // -> attackStrengthTicker = 0
   *       }
   *       this.lastItemInMainHand = main.copy();
   *     }
   *
   * and Player.attack() scales the hit by `0.2F + f*f*0.8F` where
   * `f = getAttackStrengthScale(0.5F) = clamp((ticker + 0.5) / delay, 0, 1)` and
   * `delay = getCurrentItemAttackStrengthDelay() = (1 / ATTACK_SPEED) * 20`
   * (12.5 ticks for an iron sword's 1.6 atk/s == IRON_SWORD_ATTACK_SPEED_TICKS).
   *
   * CORRECTION TO THE ISSUE'S PREMISE. The reset does NOT re-zero the meter
   * because `/clear` and `/give` are issued — there is no command hook. It is a
   * once-per-tick main-hand item-TYPE comparison, so a `/clear` + `/give` of the
   * SAME item inside ONE tick (which is exactly what arena:spawn_learner_pad
   * does) is invisible to it. On this datapack the regear therefore re-zeroes
   * the meter in exactly ONE situation: the first reset after a bot JOINS with
   * an empty main hand (air -> iron_sword). That is why the live probe deviated
   * on cycle 0 and was exact for the following 48. It is also playerdata-
   * dependent, not stochastic — a learner rejoining a PERSISTED world already
   * holding last session's sword does not take the branch at all.
   *
   * RETRACTED — an earlier revision of this comment claimed a second case, and
   * shipped it as verified fact. It is FALSE on Paper 1.21.1. Quoted so a grep
   * for the original words lands on the refutation:
   *   FALSE, RETRACTED: "the first reset after a learner death, because
   *   FALSE, RETRACTED:  PlayerList.respawn builds a NEW ServerPlayer whose
   *   FALSE, RETRACTED:  lastItemInMainHand starts at ItemStack.EMPTY"
   * PlayerList's only `new ServerPlayer` is in canPlayerLogin; BOTH respawn
   * overloads REUSE the instance through ServerPlayer.restoreFrom (the
   * CraftBukkit entity-identity patch). `lastItemInMainHand` is referenced by
   * exactly ONE class in the whole jar — Player — so neither restoreFrom nor
   * ServerPlayer.reset() clears it, and `attackStrengthTicker` (LivingEntity's
   * field) is untouched by both. With keepInventory on, the sword survives the
   * death, the first post-respawn tick sees no type change, and the ticker
   * simply carries over. The vanilla constructor initialiser IS real; on Paper
   * it just never runs a second time.
   *
   * THE REAL REASON THE ANCHOR IS UNCONDITIONAL — and why a conditional one
   * would be strictly WORSE, not merely equivalent. A conditional anchor is
   * derivable: `learner.heldItem` is readable before the reset goes out. It
   * would still be wrong, because the item comparison is NOT the only thing
   * that re-zeroes the server's meter. THE PREVIOUS EPISODE'S FINAL KILL SWING
   * ZEROES IT TOO — Paper's LivingEntity.actuallyHurt resets the attacker's
   * ticker when the damage lands, and ServerPlayer.swing resets it as well
   * (mineflayer sends arm_animation right after use_entity) — and handleReset
   * clears executor.lastSwingTick, so the bridge forgets it ever happened. The
   * reset path is not guaranteed to outlast the recovery period either: 12.5
   * SERVER ticks is 658-833 ms at this machine's measured 15-19 TPS, and the
   * live 1.269 hit shows the regear -> ack -> first step leg alone can cost a
   * single tick. So a conditional's "accurate" branch — report 1.0 in steady
   * state because heldItem is already a sword — would be OPTIMISTIC against a
   * source neither heldItem nor death-tracking can observe. The unconditional
   * anchor covers swing carryover, the join case and the respawn no-op
   * identically, on every cycle. That is why it is right, not just safe.
   *
   * heldItem could not even substitute for the item branch alone: the client is
   * never told about intra-tick inventory states. AbstractContainerMenu
   * .triggerSlotListeners diffs each slot against `lastSlots` with
   * ItemStack.matches and sends nothing when they agree (verified in the same
   * jar), so a same-tick /clear + /give of an equal stack produces NO packet,
   * and an unequal one (a worn sword replaced by a fresh one) produces a single
   * set_slot that says nothing about an item-TYPE change.
   *
   * The residual error is bounded and one-directional: the wire under-reports by
   * whatever charge the server accumulated during the reset path, costing at
   * most ceil(12.5 / ACTION_REPEAT) == 4 windows of waiting. Under-reporting
   * makes a policy wait when it could have swung; OVER-reporting is the bug
   * being fixed, and would be a standing invitation to a partial-cooldown hit.
   *
   * WHY TICK 0 AND NOT A WALL-CLOCK OFFSET FROM THE CAUSALITY BEACON. The
   * beacon IS the causally correct anchor — spawn_learner_pad emits it on the
   * line after the regear `/give`, in the same tick, and it gates the ack — so
   * anchoring the post-reset tick frame at 0 already anchors at the beacon,
   * erring conservative by exactly the beacon->ack gap. Crediting that gap back
   * explicitly was considered and rejected: it would have to convert wall-clock
   * ms into ticks at an assumed 20 TPS, and this server is measured at 15-19,
   * so the correction would err OPTIMISTIC — the one direction that re-creates
   * this defect. The gap is also small: the live 1.269-damage first hit is
   * exactly 6 * (0.2 + 0.8 * (1.5/12.5)^2), so the server's ticker read 1 when
   * that swing was processed — the whole regear -> beacon -> ack -> first step
   * path had cost a single tick.
   *
   * It also protects the probe's arithmetic. 12.5 is not a multiple of
   * ACTION_REPEAT, so the ramp first clears 1.0 at tick 16 — about 3.5 ticks
   * of overshoot past the true period, which absorbs sub-20 TPS jitter and
   * inter-step latency. Crediting 1-3 ticks back would move the first swing to
   * tick 12, i.e. right on the boundary, where a server running below 20 TPS
   * would land a scale-<1 hit and break the exact 6,6,6,2 sequence.
   *
   * NOT A SWING GATE. MacroExecutor.canSwing() is untouched and still allows an
   * ATTACK at w0. That is faithful: the server allows a weak swing there, and an
   * action silently downgraded to IDLE would be a worse lie than the one being
   * fixed. The agent is now simply told the swing is not charged.
   *
   * @returns {number} Swing progress clamped to [0, 1].
   */
  attackCooldown() {
    const lastSwingTick = this.executor !== null ? this.executor.lastSwingTick : null;
    const sinceSwing = computeAttackCooldown(
      this._currentTick,
      lastSwingTick,
      this._weaponAttackSpeedTicks,
    );
    const sinceRegear = computeAttackCooldown(
      this._currentTick,
      this._meterResetTick,
      this._weaponAttackSpeedTicks,
    );
    return sinceSwing < sinceRegear ? sinceSwing : sinceRegear;
  }

  /**
   * Server-authoritative game tick for the OUTBOUND `state.tick` field.
   *
   * Sourced from the learner bot's world age (`bot.time.age`), which Mineflayer
   * sets ONLY from the server `update_time` packet (node_modules/mineflayer/lib/
   * plugins/time.js), so it reflects the REAL server tick rate, decoupled from
   * the client-side physicsTick timer that `this._currentTick` and the swing/
   * cooldown gate ride on. The server sends `update_time` only ~once per second,
   * so this value updates coarsely (flat, then jumps ~20); the eval benchmark
   * averages it over a rolling window to recover the true rate (eval/benchmark.py
   * TickDeltaTpsProvider).
   *
   * Falls back to the internal per-step counter (`this._currentTick`) before the
   * first `update_time` arrives, and for unit-test fakes whose learner has no
   * `time`. assembleStateMsg still clamps the result to a non-negative integer.
   *
   * @returns {number} The learner's server world-age tick, or the internal counter.
   */
  _serverTick() {
    if (
      this.learner &&
      this.learner.time &&
      Number.isInteger(this.learner.time.age) &&
      this.learner.time.age >= 0
    ) {
      return this.learner.time.age;
    }
    return this._currentTick;
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

    // ONE opponent resolution for the WHOLE decision window. _opponentHandle()
    // is stateless and re-resolves from the live entity map on every call, so
    // resolving it separately at window START (the attack target) and at window
    // END (the last-seen memory, the observation) can land on two DIFFERENT
    // people in 'human' mode with challengerUsername = null: the entity map
    // changes mid-window and Object.keys hands back a different first player.
    // The agent would then swing at one person and turn toward another. Resolve
    // once here and thread that single handle through every read below. A null
    // handle is passed through as "no opponent this window" — the reads honor
    // it rather than re-resolving (only an OMITTED argument re-resolves).
    //
    // This is intra-step CONSISTENCY only, not a claim on the slot: the handle
    // dies with the window, so a challenger who disconnects is null on the very
    // next step. The first-claimant latch is T3's.
    const opponentHandle = this._opponentHandle();
    const opponentEntity = this._opponentEntity(opponentHandle);

    // 2. Begin the macro for this window (gated swing happens here, at tick 0).
    if (this.executor !== null) {
      this.executor.begin(action, {
        currentTick: windowStartTick,
        opponentEntity,
        lastSeenPosition: this._lastSeenOpponentPos,
      });
    }

    // 3. Hold for ACTION_REPEAT ticks. The wired per-bot health handlers fold
    //    every hit into this.events during the wait; we update the last-seen
    //    memory from perception as the opponent is observed.
    await this._waitTicks(ACTION_REPEAT);
    this._updateLastSeen(opponentHandle);

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
      opponent: this._snapshotOpponent(opponentHandle),
      events,
      wallDistances: this._probeWallDistances(),
      tick: this._serverTick(),
      codeVersion: resolveCodeVersion(),
    });
    this._trySend(stateMsg);
  }

  // -------------------------------------------------------------------------
  // OPPONENT HANDLE (T1) — the single seam every BEHAVIORAL opponent read goes
  // through: ATTACK's target, TURN_TO_LAST_SEEN's memory, the observation
  // snapshot, and the damage channel. Before this seam existed those sites read
  // `this.dummy` directly, which silently assumed the opponent always has a
  // Mineflayer connection of its own — false for a human challenger, who is a
  // player entity and nothing more.
  //
  // BOT-LIFECYCLE work (spawning, the reset read-back gate, quitting, the reset
  // causality beacon) deliberately does NOT go through here: it acts on the
  // dummy CONNECTION, so it stays bot-specific and no-ops for a human.
  // -------------------------------------------------------------------------

  /**
   * Whether the opponent is this pad's dummy bot rather than a human.
   *
   * Keyed on the configured MODE, not on `this.dummy` being populated: the
   * lifecycle sites that ask this must behave identically to today while the
   * dummy is merely not connected YET (pre-connect, or a failed spawn).
   *
   * @returns {boolean}
   */
  _opponentIsBot() {
    return this.opponentMode === OPPONENT_MODE_BOT;
  }

  /**
   * WHO this pad is fighting, or null when there is no opponent at all.
   *
   * `entity` may be null inside a NON-null handle: in 'bot' mode the dummy
   * exists but has not spawned yet. That distinction is load-bearing —
   * _snapshotOpponent reports a bot's health even before its entity appears.
   *
   * `healthSource` states whether health can be read at all, rather than
   * letting a caller "read" one that resolves to undefined. Mineflayer never
   * populates `entity.health` for anyone but the connected bot, so a human
   * challenger's health is 'unavailable', full stop — do not synthesize one
   * from the player entity (it is always undefined, and a fabricated reading
   * becomes phantom damage_dealt).
   *
   * @returns {{entity: object|null, isBot: boolean, username: string|null,
   *            healthSource: 'own-connection'|'unavailable'}|null}
   */
  _opponentHandle() {
    if (this.opponentMode === OPPONENT_MODE_HUMAN) {
      const entity = this._resolveChallengerEntity();
      if (entity === null) {
        // Nobody has joined yet, or the challenger left mid-match. Every call
        // site treats this exactly like "no opponent": no attack target, no
        // memory update, a zeroed opponent block on the wire.
        return null;
      }
      return {
        entity,
        isBot: false,
        // No fallback: _resolveChallengerEntity only returns an entity whose
        // `username` is already a string, so a `this.challengerUsername`
        // fallback here would stand for a state that cannot occur.
        username: entity.username,
        healthSource: OPPONENT_HEALTH_UNAVAILABLE,
      };
    }
    if (!this.dummy) {
      return null;
    }
    return {
      entity: this.dummy.entity ? this.dummy.entity : null,
      isBot: true,
      username:
        typeof this.dummy.username === 'string' ? this.dummy.username : this.config.dummyUsername,
      healthSource: OPPONENT_HEALTH_OWN_CONNECTION,
    };
  }

  /**
   * The opponent's OWN Mineflayer connection, or null when the opponent has
   * none (a human) — the only thing that can source health/death events.
   *
   * @returns {object|null}
   */
  _opponentBot() {
    return this._opponentIsBot() && this.dummy ? this.dummy : null;
  }

  /**
   * The opponent's current health, or null when it cannot be read.
   *
   * Null means "no reading", NEVER "0 health": a human challenger has no
   * connection to read from, and treating that as 0 would fabricate a kill.
   * T2 sources a human's death from the `rl_deaths` scoreboard instead.
   *
   * @returns {number|null} A finite health value, or null.
   */
  _opponentHealth() {
    const handle = this._opponentHandle();
    if (handle === null || handle.healthSource !== OPPONENT_HEALTH_OWN_CONNECTION) {
      return null;
    }
    const bot = this._opponentBot();
    return bot && typeof bot.health === 'number' && Number.isFinite(bot.health) ? bot.health : null;
  }

  /**
   * The challenger's player entity in the learner's own view, or null.
   *
   * Resolved from `learner.entities` (the same source as _scanForeignPlayers)
   * rather than the server-wide `bot.players` roster: the entity view is scoped
   * to what is actually near this pad, so a neighbouring pad's bots 512 blocks
   * away can never be mistaken for a challenger. This pad's own two bots are
   * excluded by username.
   *
   * STATELESS by design: it re-resolves on every call, so a challenger who
   * disconnects immediately reports null. The "first player claims the slot and
   * later joiners are ignored until reset" protocol is exhibition policy and
   * belongs to T3, which owns the latch on top of this primitive (as does any
   * pad-entry gating).
   *
   * @returns {object|null} A Mineflayer player entity, or null.
   */
  _resolveChallengerEntity() {
    const entities =
      this.learner && this.learner.entities && typeof this.learner.entities === 'object'
        ? this.learner.entities
        : null;
    if (entities === null) {
      return null;
    }
    const own = new Set([this.config.learnerUsername, this.config.dummyUsername]);
    for (const key of Object.keys(entities)) {
      const entity = entities[key];
      if (!entity || entity.type !== 'player') {
        continue;
      }
      const name = typeof entity.username === 'string' ? entity.username : null;
      if (name === null || own.has(name)) {
        continue;
      }
      if (this.challengerUsername !== null && name !== this.challengerUsername) {
        continue;
      }
      return entity;
    }
    return null;
  }

  /**
   * The opponent Mineflayer entity, the bot.attack target, or null.
   *
   * @param {object|null} [handle] The handle already resolved for THIS decision
   *   window (see handleStep). Only an OMITTED argument re-resolves; an explicit
   *   `null` is the legitimate "no opponent" answer and is honored as one.
   * @returns {object|null}
   */
  _opponentEntity(handle = this._opponentHandle()) {
    if (handle !== null && handle.entity) {
      return handle.entity;
    }
    return null;
  }

  /**
   * Update the last-seen opponent position from the opponent's current position.
   *
   * @param {object|null} [handle] The handle already resolved for THIS decision
   *   window, so the memory records the person the window's swing was aimed at
   *   rather than whoever the entity map happens to yield now. The POSITION is
   *   still read live off that entity here, at window end, exactly as before.
   *   Only an OMITTED argument re-resolves; an explicit `null` means "no
   *   opponent", which leaves the existing memory untouched.
   */
  _updateLastSeen(handle = this._opponentHandle()) {
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
    // T1 changed WHERE the position comes from (the opponent handle, so a human
    // challenger is tracked too) and NOTHING else. The write below stays
    // UNCONDITIONAL — see the TODO(T12) note above: it is the agent's only way
    // to re-acquire an opponent it cannot see, and gating it breaks the demo.
    const pos = handle !== null && handle.entity ? handle.entity.position : null;
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

  /**
   * Snapshot the opponent's RAW state (incl. PRIVILEGED health) for the wire.
   *
   * @param {object|null} [handle] The handle already resolved for THIS decision
   *   window, so the observation describes the same person the window's swing
   *   and last-seen memory used. Only an OMITTED argument re-resolves (that is
   *   handleReset's post-reset first observation, which is outside any window);
   *   an explicit `null` yields the zeroed opponent block.
   */
  _snapshotOpponent(handle = this._opponentHandle()) {
    const entity = handle !== null && handle.entity ? handle.entity : null;
    // Health is readable ONLY from an opponent with its own connection (T1).
    // For a human challenger the wire carries 0 here — the same zeroed block an
    // absent opponent already produces, and explicitly NOT a health reading.
    // It is deliberately not synthesized from the player entity: mineflayer
    // leaves entity.health undefined for non-self players, so a "reading" taken
    // there would be a fabricated one. The demo runs greedy with no learning,
    // so nothing downstream consumes it; win detection is T2's scoreboard.
    //
    // The predicate below stays the LOOSE `typeof` this site has always used —
    // deliberately not _opponentHealth()'s finite-only one. NOT because the two
    // differ on the wire: they cannot. assembleStateMsg passes this value
    // through finiteOr(opponent.health, 0), so a NaN/Infinity reading is
    // already 0 on the wire under either predicate. They are left un-unified
    // because unifying them is a cosmetic cleanup on a FROZEN training path,
    // and "byte-identical on the training path" is the bar for this change.
    const bot = this._opponentBot();
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

  // NOTE (T9): `_regear()` is gone. Regearing — /clear then a namespaced /give
  // per template item — now lives in the datapack's spawn_learner_pad /
  // spawn_dummy_pad macros, together with the rest of the reset template, so
  // there is exactly one place that decides what a bot holds at episode start.
  // The clear-before-give ordering it used to encode is preserved there (and is
  // load-bearing for the instant effects granted in the same tick).

  /** Tear down both bots and the transport. */
  async close() {
    if (this.learner && typeof this.learner.quit === 'function') {
      this.learner.quit();
    }
    // BOT LIFECYCLE — already a no-op for a human opponent: this.dummy is null
    // there, and a challenger's client is not ours to disconnect (T1).
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
  OPPONENT_MODE_BOT,
  OPPONENT_MODE_HUMAN,
  OPPONENT_HEALTH_OWN_CONNECTION,
  OPPONENT_HEALTH_UNAVAILABLE,
  // Pure, unit-testable logic.
  assertMacroInt,
  assertMacroUsername,
  formatSetupPadCommand,
  formatResetPadCommand,
  formatResetConfirmation,
  readbackMatchesTemplate,
  computeAttackCooldown,
  snapshotBotState,
  buildEventsBlock,
  assembleStateMsg,
  // Live gate loop + arena owner (structure for the live handshake).
  runReadbackGate,
  ArenaBots,
};
