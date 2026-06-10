// actions.js — Action macro execution + damage-event aggregation (T7b).
//
// Two responsibilities, both kept as PURE / mockable units so they are
// unit-testable with `node --test` WITHOUT a live Minecraft server:
//
//   1. MACRO MAPPING (macroToControlStates / MacroExecutor): translate a
//      discrete action integer (the frozen agent/actions.py Macro enum) into the
//      exact low-level Mineflayer calls the resolved T7b execution model
//      prescribes — raw bot.setControlState(...) for movement, raw
//      bot.attack(entity) for a SINGLE cooldown-gated swing, bot.lookAt(...) for
//      the memory-driven turn. It DELIBERATELY never touches bot.pvp.attack or a
//      pathfinder goal (those own the cooldown / drive async pursuit and would
//      swing across decision boundaries — forbidden by the contract).
//
//   2. EVENT AGGREGATION (EventAggregator): fold the per-tick damage/death events
//      observed across one ACTION_REPEAT-tick decision window into ONE
//      events block ({damage_dealt, damage_taken, i_died, opponent_died}),
//      counting every damage event EXACTLY ONCE — no drop, no double-count at the
//      window boundary. This is the deepest silent risk in the whole bridge: a
//      mis-counted hit corrupts the reward's damage anchor with no error
//      anywhere. The aggregator is a pure accumulator (no Mineflayer, no clock);
//      bot.js feeds it from the live entityHurt / health events and drains it at
//      the boundary.
//
// The macro contract is FROZEN in agent/actions.py (Macro IntEnum +
// MACRO_SEMANTICS) and the timing in agent/contract_config.py (ACTION_REPEAT=4).
// The values here mirror those exactly; a drift is a contract bug.
//
// ============================================================================
// VERIFIED HERE (node --test, NO live server — see actions.test.js):
//   TC7  EventAggregator counts a 3-hit exchange straddling a window boundary
//        as EXACTLY 3 — no drop, no double-count; hits at the first tick, the
//        last tick, and split across two windows all sum correctly; i_died /
//        opponent_died latch once per window and reset on drain.
//   - macroToControlStates maps each movement macro to the right control
//     state(s); IDLE maps to none; the mapping never references pathfinder.
//   - MacroExecutor.begin(...) sets the correct control state(s) on a mock bot;
//     MacroExecutor.end(...) clears the transient states it set.
//   - ATTACK calls bot.attack(entity) (a single swing) and respects the cooldown
//     gate: a second ATTACK within the weapon cooldown does NOT swing again.
//   - ATTACK / movement never call bot.pvp.attack or a pathfinder goal (asserted
//     against a mock bot whose pvp/pathfinder calls would be recorded).
//   - TURN_TO_LAST_SEEN calls bot.lookAt toward the stored last-seen position;
//     with no memory it is a no-op (no look, no throw).
// LIVE-ONLY (requires the Paper 1.21.1 server, per server/compat_check.md):
//   TC7b The real damage exchange — two opped bots, real entityHurt timing,
//        real swing cooldown — verifying landed hits actually move health and
//        the aggregated events match. Documented human follow-up.
// ============================================================================
//
// Owner: T7b (Environment/bridge track)

'use strict';

// ---------------------------------------------------------------------------
// Frozen macro enum. MUST match agent/actions.py Macro IntEnum exactly (any
// drift silently mis-maps actions). Mirrored here, not imported, because this is
// the Node side; the Python enum is the authority.
// ---------------------------------------------------------------------------

const Macro = Object.freeze({
  IDLE: 0,
  APPROACH: 1,
  RETREAT: 2,
  STRAFE_L: 3,
  STRAFE_R: 4,
  ATTACK: 5,
  JUMP: 6,
  TURN_TO_LAST_SEEN: 7,
});

/** Number of macros (0..7). Mirrors N_ACTIONS in agent/actions.py. */
const N_ACTIONS = Object.keys(Macro).length;

/** Inclusive action-index bounds (mirror schema.json step.action). */
const ACTION_MIN = 0;
const ACTION_MAX = N_ACTIONS - 1;

// ---------------------------------------------------------------------------
// Macro -> Mineflayer control-state(s) mapping.
//
// The movement / jump macros are pure setControlState directions. ATTACK and
// TURN_TO_LAST_SEEN are NOT control states (they are bot.attack / bot.lookAt
// calls) and IDLE is the empty set, so they map to [] here and are handled by
// the executor. This table is the single source of truth for "which keys does
// macro X hold", so the executor can both press them on begin and release
// exactly those on end.
//
// The four Mineflayer movement control-state names, frozen by Mineflayer:
//   forward / back / left / right / jump / sprint / sneak.
// We use forward/back/left/right/jump only (the macro contract).
// ---------------------------------------------------------------------------

/** Every control state the executor may hold; cleared en masse on IDLE / reset. */
const ALL_CONTROL_STATES = Object.freeze(['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']);

/**
 * Map a macro to the set of control states it HOLDS for the decision window.
 * Macros that are not movement (ATTACK, TURN_TO_LAST_SEEN) and IDLE hold no
 * control state and return an empty array.
 *
 * @param {number} macro A Macro value (0..7).
 * @returns {string[]} The control-state names to set true (possibly empty).
 * @throws {RangeError} If `macro` is not a known macro index.
 */
function macroToControlStates(macro) {
  switch (macro) {
    case Macro.IDLE:
      return [];
    case Macro.APPROACH:
      return ['forward'];
    case Macro.RETREAT:
      return ['back'];
    case Macro.STRAFE_L:
      return ['left'];
    case Macro.STRAFE_R:
      return ['right'];
    case Macro.JUMP:
      return ['jump'];
    case Macro.ATTACK:
      // Not a control state — a single bot.attack swing (see MacroExecutor).
      return [];
    case Macro.TURN_TO_LAST_SEEN:
      // Not a control state — a bot.lookAt toward the memory position.
      return [];
    default:
      throw new RangeError(
        `unknown macro ${macro}; expected an integer in [${ACTION_MIN}, ${ACTION_MAX}]`,
      );
  }
}

/** True iff the macro is the single-swing ATTACK (handled specially, gated). */
function isAttackMacro(macro) {
  return macro === Macro.ATTACK;
}

/** True iff the macro is the memory-driven look toward the last-seen opponent. */
function isTurnMacro(macro) {
  return macro === Macro.TURN_TO_LAST_SEEN;
}

// ---------------------------------------------------------------------------
// Shared timing constant — single source of truth for the iron-sword cooldown
// period so actions.js (MacroExecutor default) and bot.js (ArenaBots) never
// drift from each other. Exported so bot.js can import it rather than
// recomputing `SERVER_TPS / 1.6` independently.
// ---------------------------------------------------------------------------

/**
 * Ticks for a full iron-sword swing recharge at vanilla 1.9+ attack speed
 * (1.6 atk/s × 20 TPS == 12.5 ticks). This is the default weapon cooldown
 * period used by both MacroExecutor and ArenaBots; it can be overridden per
 * weapon at construction time.
 */
const IRON_SWORD_ATTACK_SPEED_TICKS = 20 / 1.6;

// ---------------------------------------------------------------------------
// EventAggregator — PURE per-window damage/death accumulator (THE testable core).
//
// THE EXACTLY-ONCE-AT-BOUNDARY GUARANTEE
// --------------------------------------
// The aggregator owns a single mutable accumulator for the CURRENT window:
//   { damage_dealt, damage_taken, i_died, opponent_died }.
// bot.js feeds it every damage/death event as it observes it (recordDamageDealt,
// recordDamageTaken, recordIDied, recordOpponentDied). Each call mutates ONLY
// the current accumulator. At the window boundary, bot.js calls drain():
//   1. snapshot the current accumulator into the events block to emit, then
//   2. reset the accumulator to zero.
// Because every event is folded into the accumulator the instant it is recorded,
// and drain() is the ONLY thing that both reads and clears it, each event is
// counted in EXACTLY ONE window:
//   - an event recorded BEFORE a given drain() is included in THAT drain and
//     then cleared, so it cannot appear in the next window's drain (no
//     double-count);
//   - an event recorded AFTER a drain() lands in the fresh accumulator and is
//     emitted by the NEXT drain (no drop).
// The "window boundary" is therefore defined purely by WHEN bot.js calls drain()
// relative to recordX() — it is not tick-arithmetic the aggregator has to get
// right, which is exactly why it is trivially correct and unit-testable. bot.js
// records each live event once (de-bounced at the source) and drains once per
// ACTION_REPEAT-tick window; the aggregator guarantees the fold/clear is atomic.
//
// Deaths LATCH: i_died / opponent_died are booleans OR-ed across the window, so
// repeated death signals in one window collapse to a single true, and they reset
// to false on drain (a death belongs to exactly the window it happened in).
// ---------------------------------------------------------------------------

class EventAggregator {
  constructor() {
    /** @type {{damage_dealt:number, damage_taken:number, i_died:boolean, opponent_died:boolean}} */
    this._acc = EventAggregator._zero();
  }

  /** A fresh, zeroed accumulator (the empty events block). */
  static _zero() {
    return { damage_dealt: 0, damage_taken: 0, i_died: false, opponent_died: false };
  }

  /**
   * Record damage the learner dealt to the opponent this window.
   * Non-finite or negative amounts are ignored (defensive: a bad health-delta
   * must never corrupt the reward's damage anchor).
   *
   * @param {number} amount Damage points dealt (>= 0).
   */
  recordDamageDealt(amount) {
    if (typeof amount === 'number' && Number.isFinite(amount) && amount > 0) {
      this._acc.damage_dealt += amount;
    }
  }

  /**
   * Record damage the learner took this window. See recordDamageDealt for the
   * defensive handling.
   *
   * @param {number} amount Damage points taken (>= 0).
   */
  recordDamageTaken(amount) {
    if (typeof amount === 'number' && Number.isFinite(amount) && amount > 0) {
      this._acc.damage_taken += amount;
    }
  }

  /** Latch that the learner died this window (idempotent within the window). */
  recordIDied() {
    this._acc.i_died = true;
  }

  /** Latch that the opponent died this window (idempotent within the window). */
  recordOpponentDied() {
    this._acc.opponent_died = true;
  }

  /**
   * Read the current (not-yet-drained) accumulator WITHOUT clearing it. For
   * inspection/tests; the live step path uses drain().
   *
   * @returns {{damage_dealt:number, damage_taken:number, i_died:boolean, opponent_died:boolean}}
   */
  peek() {
    return {
      damage_dealt: this._acc.damage_dealt,
      damage_taken: this._acc.damage_taken,
      i_died: this._acc.i_died,
      opponent_died: this._acc.opponent_died,
    };
  }

  /**
   * Close the current window: return its aggregated events block and reset the
   * accumulator to zero for the next window. This is the ONLY method that both
   * reads and clears the accumulator, which is what makes the count exactly-once
   * at the boundary (see the class header).
   *
   * @returns {{damage_dealt:number, damage_taken:number, i_died:boolean, opponent_died:boolean}}
   *   The events block for the window just closed.
   */
  drain() {
    const events = this.peek();
    this._acc = EventAggregator._zero();
    return events;
  }

  /**
   * Hard reset the accumulator (e.g. on a new episode), discarding any pending,
   * not-yet-drained events. Distinct from drain() in that it returns nothing —
   * a reset's partial window is intentionally thrown away, not emitted.
   */
  reset() {
    this._acc = EventAggregator._zero();
  }
}

// ---------------------------------------------------------------------------
// MacroExecutor — drives a (live or mock) Mineflayer bot through one macro for
// the ACTION_REPEAT-tick window. Stateless across windows except for the
// attack-cooldown gate, which it OWNS (the resolved cooldown-ownership decision:
// the bridge tracks last-swing-tick and gates bot.attack manually — NEVER
// bot.pvp.attack).
//
// Lifecycle per step (driven by bot.js):
//   executor.begin(macro, ctx)   // press control states / swing / look (tick 0)
//   ... ACTION_REPEAT ticks pass, events aggregated ...
//   executor.end()               // release the transient control states it set
//
// begin/end are split so the control states are HELD across the whole window and
// then cleared, matching the frozen execution model ("held for ACTION_REPEAT
// ticks, then cleared"). ATTACK is an instantaneous swing at begin (a single
// swing, gated), not a held state; JUMP is held like movement (Mineflayer
// auto-releases on landing, but we still clear it on end for determinism).
// ---------------------------------------------------------------------------

class MacroExecutor {
  /**
   * @param {object} bot A Mineflayer bot (or a mock exposing setControlState,
   *   clearControlStates, attack, lookAt).
   * @param {object} [options]
   * @param {number} [options.weaponAttackSpeedTicks] Ticks for the held weapon's
   *   swing cooldown to fully recharge (the gate period). Defaults to an iron
   *   sword's ~1.6 atk/s @ 20 TPS == 12.5 ticks.
   */
  constructor(bot, options = {}) {
    this.bot = bot;
    this.weaponAttackSpeedTicks =
      options.weaponAttackSpeedTicks !== undefined
        ? options.weaponAttackSpeedTicks
        : IRON_SWORD_ATTACK_SPEED_TICKS;

    /**
     * The set of control states currently held by the in-flight macro, so end()
     * releases exactly those (and nothing a future macro might rely on).
     * @type {string[]}
     */
    this._heldControlStates = [];

    /**
     * Tick of the learner's last ATTACK swing this episode, or null if none yet
     * (fully charged). Owned here so the gate logic and the bridge's
     * attack_cooldown read agree on one source of truth.
     * @type {number|null}
     */
    this._lastSwingTick = null;
  }

  /** The last-swing tick (for bot.js's attack_cooldown computation). */
  get lastSwingTick() {
    return this._lastSwingTick;
  }

  /** Reset the per-episode swing gate (called on reset so cooldown starts ready). */
  resetCooldown() {
    this._lastSwingTick = null;
  }

  /**
   * True iff a swing is allowed at `currentTick` given the last swing and the
   * weapon's cooldown period. The gate is the bridge's manual replacement for
   * the bot.pvp cooldown: no swing until at least one full weapon period has
   * elapsed since the last swing.
   *
   * @param {number} currentTick The tick the swing would happen on.
   * @returns {boolean}
   */
  canSwing(currentTick) {
    if (this._lastSwingTick === null) {
      return true;
    }
    if (!(this.weaponAttackSpeedTicks > 0)) {
      // Unknown/degenerate weapon speed: never block (matches computeAttackCooldown).
      return true;
    }
    return currentTick - this._lastSwingTick >= this.weaponAttackSpeedTicks;
  }

  /**
   * Begin executing `macro` for the upcoming window: press its control states,
   * or issue its single bot.attack swing (gated), or its bot.lookAt turn.
   *
   * @param {number} macro A Macro value (0..7).
   * @param {object} [ctx]
   * @param {number} [ctx.currentTick=0] The tick this window begins on (used by
   *   the ATTACK cooldown gate and to stamp the swing).
   * @param {object|null} [ctx.opponentEntity=null] The opponent Mineflayer
   *   entity, the target of a bot.attack swing. If absent, ATTACK swings nothing
   *   (no entity to hit) — but still respects the gate so timing is unaffected.
   * @param {{x:number,y:number,z:number}|null} [ctx.lastSeenPosition=null] The
   *   stored last-seen opponent world position, the target of TURN_TO_LAST_SEEN.
   *   If absent (never seen / memory expired) the turn is a no-op.
   * @returns {{swung:boolean, looked:boolean, controlStates:string[]}} A summary
   *   of what was issued (useful for the step log and for tests).
   * @throws {RangeError} If `macro` is unknown.
   */
  begin(macro, ctx = {}) {
    const currentTick = ctx.currentTick !== undefined ? ctx.currentTick : 0;
    const opponentEntity = ctx.opponentEntity !== undefined ? ctx.opponentEntity : null;
    const lastSeenPosition =
      ctx.lastSeenPosition !== undefined ? ctx.lastSeenPosition : null;

    // Validate the macro first (throws on an out-of-range index) so a bad action
    // never half-applies a control state.
    const controlStates = macroToControlStates(macro);

    let swung = false;
    let looked = false;

    if (isAttackMacro(macro)) {
      swung = this._attack(currentTick, opponentEntity);
    } else if (isTurnMacro(macro)) {
      looked = this._turnToLastSeen(lastSeenPosition);
    } else {
      // Movement / jump / idle: hold the control states for the window.
      this._pressControlStates(controlStates);
    }

    return { swung, looked, controlStates };
  }

  /**
   * End the current window: release the control states this macro held. Idempotent
   * and safe to call when nothing was held (ATTACK / TURN / IDLE).
   */
  end() {
    for (const state of this._heldControlStates) {
      this._setControlState(state, false);
    }
    this._heldControlStates = [];
  }

  /**
   * Defensive blanket clear of every control state the executor could ever hold
   * (e.g. on reset, so no key leaks across episodes). Independent of what is
   * currently tracked as held.
   */
  clearAll() {
    if (this.bot && typeof this.bot.clearControlStates === 'function') {
      this.bot.clearControlStates();
    } else {
      for (const state of ALL_CONTROL_STATES) {
        this._setControlState(state, false);
      }
    }
    this._heldControlStates = [];
  }

  // -- internals -----------------------------------------------------------

  /** Press the given control states and remember them for end(). */
  _pressControlStates(states) {
    // Clear any leftover held state from a prior macro before pressing the new
    // set, so a missed end() can never compound keys across windows.
    for (const state of this._heldControlStates) {
      if (!states.includes(state)) {
        this._setControlState(state, false);
      }
    }
    for (const state of states) {
      this._setControlState(state, true);
    }
    this._heldControlStates = states.slice();
  }

  /**
   * Issue a SINGLE cooldown-gated swing via raw bot.attack(entity). Returns
   * whether a swing actually happened. NEVER calls bot.pvp.attack.
   */
  _attack(currentTick, opponentEntity) {
    if (!this.canSwing(currentTick)) {
      // Still cooling down from the last swing — do not swing again.
      return false;
    }
    if (opponentEntity === null || opponentEntity === undefined) {
      // No target this window; consume nothing, but do not stamp a swing (a swing
      // at nothing should not start the cooldown — there was no attack).
      return false;
    }
    if (this.bot && typeof this.bot.attack === 'function') {
      this.bot.attack(opponentEntity);
      this._lastSwingTick = currentTick;
      return true;
    }
    return false;
  }

  /**
   * Turn to face the stored last-seen opponent position via bot.lookAt(pos, true)
   * (force=true bypasses interpolation). A no-op when there is no memory.
   */
  _turnToLastSeen(lastSeenPosition) {
    if (
      lastSeenPosition === null ||
      lastSeenPosition === undefined ||
      typeof lastSeenPosition.x !== 'number' ||
      typeof lastSeenPosition.y !== 'number' ||
      typeof lastSeenPosition.z !== 'number'
    ) {
      // Never seen the opponent (or memory expired): nothing to face.
      return false;
    }
    if (this.bot && typeof this.bot.lookAt === 'function') {
      this.bot.lookAt(lastSeenPosition, true);
      return true;
    }
    return false;
  }

  /** Set one control state, tolerant of a not-yet-ready bot. */
  _setControlState(state, value) {
    if (this.bot && typeof this.bot.setControlState === 'function') {
      this.bot.setControlState(state, value);
    }
  }
}

module.exports = {
  // Frozen enum + bounds (mirror agent/actions.py).
  Macro,
  N_ACTIONS,
  ACTION_MIN,
  ACTION_MAX,
  ALL_CONTROL_STATES,
  // Shared timing constant (single source of truth — imported by bot.js).
  IRON_SWORD_ATTACK_SPEED_TICKS,
  // Pure macro mapping.
  macroToControlStates,
  isAttackMacro,
  isTurnMacro,
  // Pure event aggregation (the deepest-tested unit).
  EventAggregator,
  // Macro execution against a (live or mock) bot.
  MacroExecutor,
};
