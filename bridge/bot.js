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
//   A `step` may also carry an OPPONENT action (T11b): a second MacroExecutor,
//   bound to the opponent's own connection, runs it in the SAME window and on
//   the same window-start tick, and the state reports whether its swing fired.
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

// validateInbound is the ONE implementation of the Python -> Node command
// contract; handleStep calls it instead of re-checking `action` inline (T11b).
const { BridgeServer, validateInbound } = require('./transport');
const {
  EventAggregator,
  MacroExecutor,
  Macro,
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

// ---------------------------------------------------------------------------
// HUMAN DEATH DETECTION (T2). A human challenger has no Mineflayer connection,
// so the `death` event that makes opponent_died work in 'bot' mode does not
// exist for them — and neither does a health channel (mineflayer never
// populates entity.health for non-self players). The only server-side signal
// left is the SCOREBOARD: a `deathCount` objective the server increments in
// ServerPlayer.die(), pushed to every client as a packet.
//
// The whole mechanism is 'human'-mode only. In 'bot' mode nothing here is
// wired and no command is issued, so the training path is byte-identical.
// ---------------------------------------------------------------------------

/** The `deathCount` objective a human challenger's deaths ride on. */
const RL_DEATHS_OBJECTIVE = 'rl_deaths';

/**
 * The display slot the objective is pinned to, and WHY it is not optional.
 *
 * VERIFIED AT PRIMARY SOURCE, not from memory (`javap -p -c` on
 * server/versions/1.21.1/paper-1.21.1.jar):
 * `ServerScoreboard.onScoreChanged` broadcasts ClientboundSetScorePacket ONLY
 * inside `if (this.trackedObjectives.contains(objective))`, and the sole caller
 * of `startTrackingObjective` in the whole jar is `setDisplayObjective`. So an
 * objective that is merely ADDED emits no packet to anyone, ever: without this
 * second command `/scoreboard objectives add` succeeds, the server counts the
 * deaths, and the bridge is told nothing — a silent failure of exactly the
 * shape this project keeps getting bitten by.
 *
 * `list` (the tab player-list) rather than `sidebar`/`below_name`: it is the
 * least intrusive slot on the challenger's own screen.
 */
const RL_DEATHS_DISPLAY_SLOT = 'list';

/** How long connect() waits for the server to echo the objective back. */
const RL_DEATHS_READBACK_TIMEOUT_MS = 5000;

// ---------------------------------------------------------------------------
// EXHIBITION MODE (T3) — THE FIRST-CLAIMANT LATCH.
//
// `_resolveChallengerEntity()` used to scan the learner's whole entity view and
// hand back whichever non-own player `Object.keys` yielded first — which is the
// lowest ENTITY ID, not "first to enter the pad", and is re-decided on every
// call. Two things make that unacceptable now:
//
//   1. The learner's entity view reaches far past this pad's bedrock ring, so
//      "a player is in view" is a much larger region than "a player is in the
//      arena". Somebody wandering the flat world hundreds of blocks away, or
//      standing on the terrain BELOW the pad at the same x/z, qualified.
//   2. Since T2 that is no longer merely "the agent aims at the wrong person"
//      (visible on screen, recoverable). `rl_deaths` is server-wide, so a
//      bystander who happens to resolve as the opponent and then dies TO
//      ANYTHING — fall, lava, another player — is credited as the agent's win.
//      Silent, and directly contrary to AC3.
//
// So the slot is CLAIMED: the first eligible player standing inside this pad
// takes it, later joiners are ignored, and the slot is only released by a
// reset (the operator's between-challengers command, T6). Resolution afterwards
// is BY NAME, so the claimant keeps the slot across a disconnect and nobody
// inherits it by having a lower entity ID.
//
// A pinned `challengerUsername` IS the claim: it is the operator naming the
// challenger up front, which is strictly stronger than any latch, and it
// deliberately skips the pad test so a pin works before anyone has walked in.
// ---------------------------------------------------------------------------

/**
 * The pad's occupiable interior, as OFFSETS from the pad anchor, mirroring the
 * geometry `arena:setup_pad` actually builds (see its "EXACT BOUNDS" header):
 * floor at y=63, an 8-block air column at y=64..71, and a closed bedrock ring,
 * leaving blocks x in [A.x-7, A.x+15], z in [A.z-11, A.z+11] standable.
 *
 * The bounds below are BLOCK bounds used against a player's CENTRE, so they are
 * generous by up to half a block on each face — deliberately. This test decides
 * whether somebody is close enough to be the challenger, and the cost of the
 * two errors is wildly asymmetric: half a block of slack lets an operator who
 * is hugging a wall be claimed, while half a block of strictness would refuse
 * to claim the person standing in front of the agent, with a demo audience
 * watching. Everything outside is bedrock, so the slack is unoccupiable anyway.
 *
 * The y band is what keeps the flat world's OWN ground out: the pads float at
 * y=62..71 while the superflat terrain sits at y=-61..-64, so an x/z-only test
 * would claim somebody standing far below the arena.
 */
const PAD_INTERIOR_BOUNDS = Object.freeze({
  minDx: -7,
  maxDx: 16,
  minDz: -11,
  maxDz: 12,
  // Feet level inside the pad is y=64..71 (the air column). One block of slack
  // below (the floor) and one above the ring for a jump at the top.
  minY: 63,
  maxY: 72,
});

/** Feet level of a pad's floor — the y `arena:setup_pad` positions from. */
const PAD_FEET_Y = 64;

/**
 * Radius of the datapack's loose-entity sweep. Mirrors the `distance=..64` in
 * BOTH `arena:setup_pad` and `arena:reset_pad`, which are deliberately equal so
 * no band of space is reached by one sweep and missed by the other.
 */
const PAD_SWEEP_RADIUS = 64;

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
 * The two commands that make a human challenger's deaths observable (T2), in
 * the order they must be issued.
 *
 * BOTH ARE REQUIRED. `add` creates the objective and makes the server count
 * deaths; `setdisplay` is what puts it in `ServerScoreboard.trackedObjectives`,
 * which is the gate on every ClientboundSetScorePacket (see
 * RL_DEATHS_DISPLAY_SLOT for the decompiled proof). Issuing only the first is
 * the silent-failure case: no error anywhere, and no death ever reported.
 *
 * RE-ISSUING IS SAFE AND EXPECTED. On a second bridge run the objective already
 * exists and the server answers `add` with "An objective already exists by that
 * name" — a chat error to the opped bot, a no-op server-side. The bridge never
 * parses that reply (this file scrapes no chat at all), so the failure cannot
 * wedge anything; `setdisplay` on the following line still re-pins the display.
 *
 * No macro, no `$`-substitution, no user-supplied text: the objective name and
 * slot are module constants, so there is nothing here to validate.
 *
 * @returns {string[]} Chat-ready commands, in issue order.
 */
function formatDeathObjectiveCommands() {
  return [
    `/scoreboard objectives add ${RL_DEATHS_OBJECTIVE} deathCount`,
    `/scoreboard objectives setdisplay ${RL_DEATHS_DISPLAY_SLOT} ${RL_DEATHS_OBJECTIVE}`,
  ];
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

/**
 * Whether a world position lies inside THIS pad's occupiable interior.
 *
 * PURE, and the gate on the first-claimant latch: "in the learner's entity
 * view" is a far larger region than "in the arena", and since T2 a bystander
 * who resolves as the opponent and then dies to anything is credited as the
 * agent's win. Only somebody actually standing in the pad may claim the slot.
 *
 * Defensive about its input on purpose: a mineflayer entity that has been seen
 * but not yet positioned carries no usable `position`, and an unreadable
 * position must read as "not in the pad" (refuse the claim) rather than throw
 * inside a decision window or, worse, claim by accident.
 *
 * @param {{x:number,y:number,z:number}|null|undefined} position A world position.
 * @param {{x:number,z:number}} anchor This pad's anchor (the learner spawn CELL).
 * @returns {boolean} True only for a position provably inside the pad.
 */
function isInsidePad(position, anchor) {
  if (position === null || typeof position !== 'object' || !anchor) {
    return false;
  }
  const { x, y, z } = position;
  if (
    typeof x !== 'number' ||
    !Number.isFinite(x) ||
    typeof y !== 'number' ||
    !Number.isFinite(y) ||
    typeof z !== 'number' ||
    !Number.isFinite(z)
  ) {
    return false;
  }
  const dx = x - anchor.x;
  const dz = z - anchor.z;
  return (
    dx >= PAD_INTERIOR_BOUNDS.minDx &&
    dx <= PAD_INTERIOR_BOUNDS.maxDx &&
    dz >= PAD_INTERIOR_BOUNDS.minDz &&
    dz <= PAD_INTERIOR_BOUNDS.maxDz &&
    y >= PAD_INTERIOR_BOUNDS.minY &&
    y <= PAD_INTERIOR_BOUNDS.maxY
  );
}

/**
 * The per-episode reset commands for a pad whose opponent is a HUMAN (T3).
 *
 * WHY THIS EXISTS RATHER THAN `arena:reset_pad`. That macro's third line is
 * `$function arena:spawn_dummy_pad {...dummy:"$(dummy)"...}`, and every one of
 * that file's eleven lines addresses `$(dummy)` as a selector. In exhibition
 * mode no dummy bot is connected, so all eleven fail to find a player and the
 * server prints eleven errors PER RESET into the console the operator is
 * watching during a demo. They are runtime selector no-ops, not a macro abort —
 * the `$`-substitutions stay syntactically valid, so the function instantiates
 * and its other lines still run — but they are noise that hides real problems,
 * and the reset is asking the server to do work for a bot that does not exist.
 *
 * The `dummy` key cannot simply be dropped: a macro function errors if a
 * referenced key is absent, which WOULD be the silent whole-function abort. Nor
 * can another name be substituted — the learner's would make `spawn_dummy_pad`
 * clear, teleport and knockback-pin the AGENT. So human mode issues the two
 * lines of `arena:reset_pad` that apply to it, directly:
 *
 *   1. the loose-entity sweep, verbatim (same position, same radius);
 *   2. `arena:spawn_learner_pad`, with the same arguments reset_pad forwards.
 *
 * The causality chain is intact: `spawn_learner_pad`'s last line is the
 * nonce-stamped learner beacon, and `_resetWasConfirmed('dummy')` already
 * auto-confirms when there is no dummy connection. The read-back gate is
 * already keyed on the opponent MODE, so it skips the dummy gate here too.
 *
 * These two strings DUPLICATE lines of `reset_pad.mcfunction`, which is a real
 * drift hazard — pinned by a test that reads the committed datapack and
 * compares, so an edit there fails here instead of in a live exhibition.
 *
 * @param {{x:number, z:number, learner:string, nonce:number}} args
 * @returns {string[]} Chat-ready commands, in issue order.
 */
function formatHumanResetCommands(args) {
  const spec = args || {};
  const x = assertMacroInt(spec.x, 'pad anchor x');
  const z = assertMacroInt(spec.z, 'pad anchor z');
  const learner = assertMacroUsername(spec.learner, 'learner username');
  const nonce = assertMacroInt(spec.nonce, 'reset nonce');
  return [
    `/execute positioned ${x} ${PAD_FEET_Y} ${z} run kill ` +
      `@e[type=!minecraft:player,distance=..${PAD_SWEEP_RADIUS}]`,
    `/function arena:spawn_learner_pad {x:${x},z:${z},learner:"${learner}",nonce:${nonce}}`,
  ];
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
 * @param {boolean|null} [parts.oppActionExecuted] THE SWING REPORT (T11b): did
 *   this window's `opp_action` take effect? A boolean puts the optional
 *   `opp_action_executed` key on the wire; anything else (absent, null) OMITS
 *   the key entirely, which is what keeps a step carrying no `opp_action` —
 *   every M1/M2 stationary-dummy step — byte-identical to the pre-T11b line.
 * @returns {object} A schema-valid `state` message (validateOutbound accepts it).
 */
function assembleStateMsg(parts) {
  const self = parts.self || {};
  const opponent = parts.opponent || {};
  const wall = Array.isArray(parts.wallDistances) ? parts.wallDistances : [];
  const msg = {
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
  // THE SWING REPORT (T11b, schema.md "the swing report"). Appended LAST and
  // only for a real boolean: the schema declares it optional, Python reads an
  // absent field as "no opp_action was sent, assume any swing fired", and a key
  // emitted unconditionally would change every dummy-path line on the wire.
  if (parts.oppActionExecuted === true || parts.oppActionExecuted === false) {
    msg.opp_action_executed = parts.oppActionExecuted;
  }
  return msg;
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
   *   - deathObjectiveTimeoutMs: budget for the `rl_deaths` read-back (T2).
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

    // THE OPPONENT'S EXECUTOR (T11b) — the second macro executor, bound to the
    // opponent's OWN bot connection so a Python opponent policy can drive it
    // through the same 8 frozen macros as the learner.
    //
    // CREATED LAZILY, on the first `step` that actually carries an `opp_action`
    // (see _bindOpponentExecutor). That laziness is the M2 guarantee, not an
    // optimization: the stationary-dummy path never sends the field, so this
    // stays null for a whole training run and NOTHING — not a control-state
    // write, not the reset's re-arm below — ever touches the dummy. Byte-
    // identical M2 behavior by construction rather than by inspection.
    this.opponentExecutor = deps.opponentExecutor || null;

    // Bound handler references retained so wireDamageEvents() can remove them
    // before re-adding on a reconnect/re-wire (W1a idempotency). Each property
    // is null until the first wire.
    this._boundOnSelfHealth = null;
    this._boundOnOpponentHealth = null;
    this._boundOnLearnerDeath = null;
    this._boundOnDummyDeath = null;
    this._boundOnLearnerMessage = null;
    this._boundOnDummyMessage = null;

    // HUMAN DEATH DETECTION (T2). All 'human'-mode only; inert in 'bot' mode.
    //
    // The scoreboard packet feed is raw (`learner._client`), so the bound
    // handlers and the client they were attached to are retained for the same
    // W1a off-before-on idempotency the bot handlers above use.
    this._deathScoreClient = null;
    this._boundOnScoreboardScore = null;
    this._boundOnResetScore = null;
    this._boundOnScoreboardObjective = null;
    this._boundOnScoreboardDisplay = null;

    /**
     * @type {Map<string, number>} Scoreboard holder name -> last observed
     * `rl_deaths` value. An INCREASE against this map is a death.
     *
     * MUST OUTLIVE EVERY RESET AND EVERY RECONNECT — see handleReset. It is
     * what makes a re-sent score IDEMPOTENT, and the server does re-send whole
     * batches at moments the bridge does not control. Both were read off the
     * pinned jar rather than assumed (`javap -p -c` on
     * server/versions/1.21.1/paper-1.21.1.jar — the PATCHED server jar, which is
     * the only one carrying net/minecraft classes; server/paper-1.21.1-133.jar
     * is the Paperclip bundler and holds nothing decompilable):
     *
     *   - EVERY JOIN, and this is the load-bearing one. PlayerList
     *     .placeNewPlayer calls updateEntireScoreboard, which for each objective
     *     currently pinned to a display slot pushes that objective's whole
     *     getStartTrackingPackets list — SetObjective, SetDisplayObjective, and
     *     ONE SetScore per existing entry — to the arriving player alone. So a
     *     reconnecting learner is handed the full death history of everyone on
     *     the server, on a socket whose listeners are already armed.
     *   - `setdisplay`, but ONLY when the objective was not already tracked:
     *     ServerScoreboard.setDisplayObjective broadcasts a bare
     *     SetDisplayObjective when it was, and calls startTrackingObjective —
     *     the same full list, to everyone — when it was not. On the common
     *     fresh path the objective was created moments earlier by `objectives
     *     add` and has no scores yet, so that list replays nothing. It has
     *     teeth when scores already existed, e.g. a server restart that
     *     reloaded them from scoreboard.dat with no display slot pinned.
     *
     * Drop the map and each of those replays reads as a fresh climb from zero,
     * handing the agent a win per historical death — the fabricated-kill
     * failure this whole path is built to avoid.
     */
    this._deathScores = new Map();

    // Whether the server has echoed the objective back (see
    // _verifyDeathObjective). Until then score packets are BASELINE-ONLY: the
    // burst `setdisplay` can trigger replays every pre-existing score (see
    // _deathScores above for exactly when it does), and a challenger who died
    // in an earlier match would otherwise be reported dead the instant the
    // bridge boots.
    this._deathObjectiveReady = false;
    this._deathObjectiveSeen = false;

    // DEATH-ATTRIBUTION MEMORY, and deliberately nothing more. `rl_deaths` is
    // server-wide, so a score packet only says "somebody died"; this is how the
    // bridge knows whether that somebody was the opponent. It is written from
    // the handle handleStep already resolves, so a death arriving while the
    // challenger's entity is momentarily gone is still attributed correctly.
    //
    // NOT the first-claimant latch (T3's): it claims nothing, blocks no
    // joiner, and is overwritten by whoever the next window resolves.
    this._challengerDeathName = null;

    // THE FIRST-CLAIMANT LATCH (T3). The username that has CLAIMED this pad's
    // challenger slot, or null while the slot is free. See the EXHIBITION MODE
    // block near the top of this file for why a stateless resolve is unsafe.
    //
    // Written in exactly ONE place — _claimChallenger(), called once per
    // decision window from handleStep — and released in exactly one place, the
    // tail of handleReset. Keeping the claim out of _resolveChallengerEntity()
    // is load-bearing, not tidiness: that resolver is reached from the
    // SCOREBOARD PACKET HANDLER (via _isChallengerName's live-resolve
    // fallback), so a resolver that claimed would let a death packet itself
    // decide who the challenger was — the exact fabricated-win path the latch
    // exists to close.
    //
    // A pinned this.challengerUsername outranks it; see _challengerSlot().
    this._claimedChallenger = null;

    // Edge-trigger state for the exhibition's log-only status line. The wire
    // has no slot for a status string (`state` is additionalProperties:false on
    // both validators and the env blocks on exactly one `state` per `step`), so
    // "waiting for a challenger" goes to the bridge log — and must be logged on
    // TRANSITIONS only. handleStep runs five times a second; a per-window line
    // would bury the reset diagnostics the operator actually needs.
    //
    // null == nothing logged yet, so the first window always states the truth.
    this._challengerPresentLogged = null;

    // Names already reported as "seen, but not in the pad", so the same distant
    // player is named once rather than five times a second. Cleared with the
    // claim on every reset, so it can never outgrow the player list.
    this._outOfPadReported = new Set();

    // Bounded wait for the objective read-back. Injectable so a unit test can
    // drive the unconfirmed path without burning 5 s of wall clock.
    this._deathObjectiveTimeoutMs =
      typeof deps.deathObjectiveTimeoutMs === 'number'
        ? deps.deathObjectiveTimeoutMs
        : RL_DEATHS_READBACK_TIMEOUT_MS;

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

    // HUMAN DEATH DETECTION (T2), 'human' mode only. Issued AFTER
    // wireDamageEvents() above so the listeners are already attached when the
    // server answers, and AWAITED rather than left floating: an unawaited
    // promise is process-fatal in this codebase, and its own bounded wait is
    // what separates the replay of past deaths from a live one. It resolves
    // without throwing on every path, so a scoreboard problem cannot fail
    // connect() — it is reported loudly instead.
    if (!this._opponentIsBot()) {
      await this._verifyDeathObjective();
    }
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

    // The HUMAN half of the same channel (T2). Wired here, alongside the bot
    // half, so both are (re-)established together on a reconnect, and BEFORE
    // connect() issues the scoreboard commands — the reply to `setdisplay` is a
    // packet burst that must not race an unwired listener.
    this._wireDeathScoreboard();
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
   * Every command this reset issues, in order.
   *
   * ONE command in 'bot' mode — byte-identical to what handleReset chatted
   * before this seam existed, so the training path cannot regress. In 'human'
   * mode the dummy half of `arena:reset_pad` would address a bot that is not
   * connected and print an error per line; see formatHumanResetCommands.
   *
   * @param {number} [nonce] Defaults to the currently armed nonce.
   * @returns {string[]} Chat-ready commands, in issue order.
   */
  _resetCommands(nonce = this._resetConfirm.nonce) {
    if (this._opponentIsBot()) {
      return [this._resetPadCommand(nonce)];
    }
    return formatHumanResetCommands({
      x: this.padOrigin.x,
      z: this.padOrigin.z,
      learner: this.config.learnerUsername,
      nonce,
    });
  }

  /**
   * Log any player in the learner's entity view that is not one of THIS pad's
   * two bots — nor, in exhibition mode, its claimed challenger (T12's cross-pad
   * isolation evidence, AC13).
   *
   * `dummy.on('health')` records a health DROP with no attacker attribution, so
   * a learner that reached a neighbouring pad would silently credit its damage
   * to that pad's policy. Walls and ≥512-block spacing make that impossible by
   * construction; this scan is the observable that PROVES it, emitted on the
   * bridge's stderr (never on the frozen wire) once per reset.
   *
   * THE CHALLENGER IS NOT CONTAMINATION (T3). A human opponent is a player in
   * the learner's view by definition, so without this exclusion every single
   * exhibition reset would name them here — and eval/benchmark.py reads this
   * line as cross-pad contamination evidence, so a perfectly clean demo would
   * read as a compromised run. The exclusion is deliberately narrow: only the
   * CLAIMED (or pinned) challenger, so a second person in the pad still shows
   * up, which is exactly the one-challenger-at-a-time evidence an operator
   * wants. A challenger standing in the pad before anything has claimed them
   * (only possible on the first reset of an exhibition, since the claim is
   * released here at the end of every later one) is likewise still named —
   * truthfully: at that instant they are an unclaimed player, not the opponent.
   *
   * @returns {string[]} Foreign usernames seen, in first-seen order.
   */
  _scanForeignPlayers() {
    const own = new Set([this.config.learnerUsername, this.config.dummyUsername]);
    // MODE-GATED, like every other exhibition path in this file. A challenger
    // name that somehow reached a 'bot'-mode pad (run.js refuses the
    // combination, but this object can be constructed directly) must not
    // quietly excuse a real player from the training path's contamination
    // evidence — that is the one reader of this line who cannot afford a hole.
    const challenger = this._opponentIsBot() ? null : this._challengerSlot();
    if (challenger !== null) {
      own.add(challenger);
    }
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
    // ONE command in 'bot' mode (unchanged); the dummy-free pair in 'human'
    // mode, where `arena:reset_pad`'s dummy half would address a bot that is
    // not connected. See _resetCommands / formatHumanResetCommands.
    for (const command of this._resetCommands(epoch)) {
      this._sendCommand(this.learner, command);
    }

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
    // THE OPPONENT'S GATE IS RE-ARMED ON THE SAME BOUNDARY (T11b). _currentTick
    // goes back to 0 below while a swing from the previous episode is still
    // stamped at (say) tick 96, so an un-re-armed gate would compute
    // `0 - 96 >= 12.5` as false and block the opponent's swing for ~27 windows
    // of the NEW episode. Python's shadow meter has no way to see that: it
    // clears _opp_last_swing_window on reset (mc_pvp_env.py, "the bridge calls
    // executor.resetCooldown() on every reset") and would read 1.0 throughout,
    // so ScriptedBot would return ATTACK every window, every window would report
    // swung=false, no stamp would ever land, and the opponent would mash ATTACK
    // for the rest of the episode instead of strafing. Null until an opp_action
    // has actually been driven, so a dummy-path reset still touches nothing.
    if (this.opponentExecutor !== null) {
      this.opponentExecutor.resetCooldown();
      this.opponentExecutor.clearAll();
    }
    this._currentTick = 0;
    this._lastSeenOpponentPos = null;
    // A new match may be a new challenger, so the death-ATTRIBUTION memory is
    // dropped alongside the last-seen memory; the episode's first step rewrites
    // it (handleStep takes the claim and notes the identity in one synchronous
    // prologue).
    //
    // WHAT THAT COSTS, stated because the first-claimant latch changed it: from
    // this handler's ack until that first step, _isChallengerName has neither a
    // memory nor a claim to match on — the slot is released at this handler's
    // tail — so its live-resolve fallback answers false and a challenger death
    // landing in that gap is DROPPED rather than credited. That gap is a few
    // milliseconds, and dropping is the conservative direction: a missed win,
    // never a fabricated one. See _isChallengerName for the full argument.
    //
    // DO NOT ADD `this._deathScores.clear()` HERE. The baseline map is a
    // different thing and must survive every reset: score packets are
    // edge-triggered, so after a reset the challenger's NEXT `rl_deaths` packet
    // is their death. Re-baselining here would read that packet as a first
    // sighting and silently swallow every win in the exhibition. Pinned by a
    // test ("a death after a reset is still reported").
    this._challengerDeathName = null;
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

    // RELEASE THE CHALLENGER SLOT (T3) — "later joiners are ignored UNTIL
    // RESET". This is the between-challengers reset the operator runs (T6);
    // arming the next challenger is the whole point of it.
    //
    // AFTER _scanForeignPlayers, deliberately: clearing first would blank the
    // exclusion the scan just gained and log the outgoing challenger as
    // cross-pad contamination on their way out — the bug this task fixes. No
    // step can interleave between the two (the transport serves one request at
    // a time), so "until reset" is observably identical either way.
    //
    // WHICH RESETS REACH THIS LINE, stated precisely — the loose version of
    // this comment ("a failed reset never releases the slot") was wrong in one
    // of its three cases and is not worth shipping in a file that already
    // carries retractions:
    //   - a STALE-EPOCH handler returns inside the try above and never gets
    //     here, so a superseded reset cannot release a slot the live reset owns;
    //   - a THROWN gate propagates past this tail, so a reset that failed
    //     outright leaves the match as it found it;
    //   - an ok:FALSE reset (both gates polled, neither matched, nothing threw)
    //     DOES get here and DOES release. Deliberately: the env answers ok:false
    //     by retrying the whole reset, and that retry re-claims on its first
    //     step; a second ok:false ends the session instead of continuing. So a
    //     released slot there is never observed by a live match.
    this._releaseChallengerSlot();

    // The frozen reset reply is TWO messages, not one: `state` doubles as the
    // post-reset first observation (schema.md), and the env's reset() blocks on
    // _recv_state() right after an ok:true ack — without this send the env
    // waits out its full recv timeout and tears the connection down. Only on
    // ok:true: after ok:false the env immediately retries the reset, and a
    // stray state would desync its request/reply stream. Skip it if the ack
    // didn't go out (the client disconnected during the gate): a state with no
    // matching reset_ack would desync the env's request/reply stream too.
    //
    // ONE FRAME WITH A ZEROED OPPONENT, BY CONSTRUCTION — not a bug (T3). The
    // argument-less _snapshotOpponent() below re-resolves through
    // _opponentHandle(), and the _releaseChallengerSlot() above just emptied the
    // slot, so in 'human' mode this observation carries no opponent even while
    // the challenger is standing in the pad. The next step's _claimChallenger()
    // re-claims and the block populates from then on. Exhibition-only ('bot'
    // mode resolves to the dummy and is unaffected) and cosmetic under a greedy
    // no-learning policy, but it does feed one distorted prev_obs into the first
    // step's reward computation.
    //
    // DO NOT reorder the claim/release to "fix" it: releasing AFTER
    // _scanForeignPlayers is load-bearing (see the block above) and pinned by
    // the test 'the exhibition challenger is NOT logged as cross-pad
    // contamination', which drives the real handleReset for exactly that reason.
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
   *   1. validate the whole message against the frozen inbound contract;
   *   2. executor.begin(macro): press control states / single gated swing / look,
   *      and the OPPONENT's executor likewise for an `opp_action` (T11b);
   *   3. wait ACTION_REPEAT ticks while the wired handlers feed the aggregator;
   *   4. executor.end() on both: release the transient control states;
   *   5. drain the aggregator (exactly-once at this boundary) and snapshot both
   *      bots, then send the assembled `state` (+ the swing report).
   *
   * @param {{type:'step', action:number, opp_action?:number|null}} msg
   */
  async handleStep(msg) {
    // 1. INBOUND VALIDATION (T11b). ONE implementation of the command contract:
    // transport.validateInbound is the Node binding of schema.json's step branch
    // and already checks `action` AND the optional `opp_action` by exactly the
    // same rule. An inline guard here used to duplicate the `action` half, which
    // left `opp_action` — the wire's newest field — checked by nothing that ever
    // runs on the receive path. Do not reintroduce a second copy: extend
    // validateStep in transport.js instead.
    //
    // The failure MODE is deliberately unchanged: report on the transport's
    // error channel and drop the step (no `state` reply), rather than throwing
    // out of the handler. _handleMessage's .catch would turn a throw into the
    // same emit, but only when handleStep is reached through wireTransport.
    try {
      validateInbound(msg);
    } catch (err) {
      this.transport.emit('error', err);
      return;
    }
    const action = msg.action;
    // Absent and explicit null are the SAME thing on this field: "the opponent
    // takes no action this window" (schema.md). Never a zeroth macro.
    const oppAction =
      msg.opp_action === undefined || msg.opp_action === null ? null : msg.opp_action;

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
    // This is intra-step CONSISTENCY only. WHO may be resolved at all is the
    // first-claimant latch's decision (T3), taken here — once per window, and
    // in this one place, so no packet handler can ever claim the slot.
    this._claimChallenger();
    const opponentHandle = this._opponentHandle();
    const opponentEntity = this._opponentEntity(opponentHandle);
    // Remember who that is, so a death packet arriving later this window (or
    // between windows) can be attributed even if their entity has gone. No-op
    // in 'bot' mode — the dummy's own `death` event needs no attribution (T2).
    this._noteChallengerIdentity(opponentHandle);
    // "Waiting for a challenger" has no wire slot; it goes to the log, on the
    // transition only (this runs five times a second).
    this._logChallengerPresence(opponentHandle);

    // HOLD IDLE WITH NO CHALLENGER (T3, exhibition only). A null handle already
    // makes ATTACK a no-op — the executor has nothing to swing at — but the
    // movement macros do not need an opponent to run, so an agent left stepping
    // its policy against an empty pad would jog and strafe around it between
    // challengers, on a projector. The action is overridden, NOT rejected: the
    // env still gets exactly one `state` per `step` and nothing desyncs.
    //
    // Keyed on the opponent MODE, so 'bot' mode is untouched: there a missing
    // dummy must keep behaving exactly as it does today (the M2 stationary-
    // dummy path never inspects the handle before executing).
    const effectiveAction =
      !this._opponentIsBot() && opponentHandle === null ? Macro.IDLE : action;

    // 2. Begin the macro for this window (gated swing happens here, at tick 0).
    if (this.executor !== null) {
      this.executor.begin(effectiveAction, {
        currentTick: windowStartTick,
        opponentEntity,
        lastSeenPosition: this._lastSeenOpponentPos,
      });
    }

    // 2b. THE OPPONENT'S MACRO (T11b), in this SAME decision window and on this
    // SAME windowStartTick. The two executors therefore share one clock: one
    // `step` == one decision window == ACTION_REPEAT gate ticks for BOTH sides,
    // which is the invariant Python's shadow swing meter reconstructs by
    // counting decision windows (schema.md "the swing report"). Anything that
    // advanced the opponent's gate by a different amount — stamping the swing
    // with the post-advance tick, or beginning twice in one window — desyncs
    // that meter and degenerates the opponent into mashing ATTACK.
    //
    // Resolved ONCE into a local so begin/end are guaranteed to act on the same
    // executor even if the opponent connection changes mid-window.
    const opponentExecutor = oppAction === null ? null : this._bindOpponentExecutor();
    // The swing report: null == "no opp_action this window" == the key is left
    // off the wire. A REQUESTED action that could not be driven at all (no
    // opponent bot — a human cannot be puppeted) reports false rather than
    // nothing: it is silently ignored in the sense of never throwing, but the
    // honest answer to "did it execute?" is no, and an omitted key would tell
    // Python to assume the swing fired and drain a meter that never charged.
    let oppActionExecuted = null;
    if (oppAction !== null) {
      oppActionExecuted =
        opponentExecutor === null
          ? false
          : opponentExecutor.begin(oppAction, this._opponentMacroContext(windowStartTick)).swung;
    }

    // 3. Hold for ACTION_REPEAT ticks. The wired per-bot health handlers fold
    //    every hit into this.events during the wait; we update the last-seen
    //    memory from perception as the opponent is observed.
    await this._waitTicks(ACTION_REPEAT);
    this._updateLastSeen(opponentHandle);

    // 4. Release the transient control states held for the window — the
    //    opponent's on the same boundary as the learner's, so a movement macro
    //    is held for exactly the window that pressed it and never leaks into
    //    the next one.
    if (this.executor !== null) {
      this.executor.end();
    }
    if (opponentExecutor !== null) {
      opponentExecutor.end();
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
      oppActionExecuted,
    });
    this._trySend(stateMsg);
  }

  /**
   * The opponent's macro executor, or null when the opponent cannot be driven.
   *
   * Bound to the opponent's OWN Mineflayer connection, because that is what a
   * macro acts on: control states, bot.attack and bot.lookAt all need a
   * connection, and _opponentHandle()'s `entity` is a view, not a puppet. In
   * 'human' mode there is no such connection — the challenger is a person on
   * their own client — so this returns null and the caller silently ignores the
   * `opp_action` (plan Error Handling: "opp_action for an absent opponent —
   * silently ignored, never throws").
   *
   * CREATED ON FIRST USE, and only from the step path (see the constructor):
   * a run that never sends `opp_action` never creates one, so the M2 dummy path
   * cannot be perturbed by code that exists for the opponent-acts path.
   *
   * @returns {object|null} A MacroExecutor bound to the opponent bot, or null.
   */
  _bindOpponentExecutor() {
    if (this.opponentExecutor !== null) {
      return this.opponentExecutor;
    }
    const bot = this._opponentBot();
    if (bot === null) {
      return null;
    }
    // THE DEFAULT WEAPON PERIOD, DELIBERATELY. No options object: the default
    // IS the contract. Python's shadow tracker hard-codes
    // OPPONENT_ATTACK_SPEED_TICKS = SERVER_TPS / 1.6 (env/mc_pvp_env.py) to
    // mirror MacroExecutor's own IRON_SWORD_ATTACK_SPEED_TICKS default, and
    // nothing on either side would catch the two drifting apart — the opponent
    // would simply stop attacking, or flail. Note the dummy is BARE-HANDED
    // (spawn_dummy_pad.mcfunction runs $clear with no $give): do NOT "correct"
    // this to a bare-hand speed. The two sides must agree, and the agreed value
    // is the default. The learner's this._weaponAttackSpeedTicks is
    // deliberately NOT reused here — that is the LEARNER's weapon, and passing
    // it would silently re-point this at whatever a future task sets it to.
    this.opponentExecutor = new MacroExecutor(bot);
    return this.opponentExecutor;
  }

  /**
   * The macro context for the OPPONENT's executor — the mirror image of the
   * learner's: whom the opponent swings at, and where it turns.
   *
   * Both point at the LEARNER, read live off its entity. The turn target
   * mirrors _updateLastSeen's frozen unconditional write (perfect tracking) so
   * TURN_TO_LAST_SEEN is not a silent no-op for the opponent the way a null
   * memory would make it; the omniscient ScriptedBot never needs stale memory
   * anyway (`can_see_target` is always true for it). The position is SNAPSHOT
   * via clone() for the same reason the learner's memory is: bot.lookAt needs a
   * real Vec3 (a plain object made it throw, and the unhandled rejection killed
   * the bridge mid-episode) and a live vector would keep moving under it.
   *
   * @param {number} currentTick The tick this window begins on — the SAME
   *   windowStartTick the learner's executor gets, so both gates ride one clock.
   * @returns {{currentTick:number, opponentEntity:object|null,
   *            lastSeenPosition:object|null}}
   */
  _opponentMacroContext(currentTick) {
    const learnerEntity = this.learner && this.learner.entity ? this.learner.entity : null;
    const pos = learnerEntity ? learnerEntity.position : null;
    let lastSeenPosition = null;
    if (pos && typeof pos.x === 'number' && typeof pos.y === 'number' && typeof pos.z === 'number') {
      lastSeenPosition =
        typeof pos.clone === 'function' ? pos.clone() : { x: pos.x, y: pos.y, z: pos.z };
    }
    return { currentTick, opponentEntity: learnerEntity, lastSeenPosition };
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
   * WHO currently holds this pad's challenger slot, or null while it is free.
   *
   * A pinned `challengerUsername` outranks the latch and needs no claim: it is
   * the operator naming the challenger up front — stronger than "whoever walked
   * in first" — and it must work with an empty entity view, e.g. attributing a
   * death that lands before the named person is anywhere near the pad.
   *
   * @returns {string|null} A username, or null.
   */
  _challengerSlot() {
    if (this.challengerUsername !== null) {
      return this.challengerUsername;
    }
    return this._claimedChallenger;
  }

  /**
   * The learner's player-entity view, or null when there is none.
   *
   * `learner.entities` (the same source as _scanForeignPlayers) rather than the
   * server-wide `bot.players` roster: the entity view is scoped to what is
   * actually near this pad, so a neighbouring pad's bots 512 blocks away are
   * not even candidates.
   *
   * @returns {object|null}
   */
  _learnerEntities() {
    return this.learner && this.learner.entities && typeof this.learner.entities === 'object'
      ? this.learner.entities
      : null;
  }

  /**
   * THE FIRST-CLAIMANT LATCH (T3). If the challenger slot is free, give it to
   * the first eligible player standing INSIDE this pad.
   *
   * Called from exactly one place — once per decision window, at the top of
   * handleStep — and deliberately NOT from _resolveChallengerEntity(), which is
   * reachable from the scoreboard packet handler. See _claimedChallenger.
   *
   * "Eligible" is: a player entity, not one of this pad's own two bots, and
   * inside the pad's bedrock ring. That last test is the one that matters most
   * (see the EXHIBITION MODE block at the top of this file): the entity view
   * extends far past the arena, and since T2 an accidental claim on a distant
   * bystander turns their next death — to anything at all — into a win the
   * agent never earned.
   *
   * A pinned username is already the claim, so this no-ops there.
   */
  _claimChallenger() {
    if (this._opponentIsBot() || this._challengerSlot() !== null) {
      // 'bot' mode has no challenger; a taken slot ignores later joiners until
      // the next reset releases it (the one-challenger-at-a-time protocol).
      return;
    }
    const entities = this._learnerEntities();
    if (entities === null) {
      return;
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
      if (!isInsidePad(entity.position, this.padOrigin)) {
        // Named ONCE per reset, not once per window: without this line an
        // exhibition where the challenger joined at world spawn instead of the
        // pad looks identical to one where nobody joined at all — the agent
        // idles and nothing says why.
        if (!this._outOfPadReported.has(name)) {
          this._outOfPadReported.add(name);
          console.error(
            `[bridge] pad ${this.padIndex} challenger_outside_pad ${name} — seen in the ` +
              `learner's view but not inside the pad at anchor ${this.padOrigin.x},` +
              `${this.padOrigin.z}; they must be in the arena to be claimed`,
          );
        }
        continue;
      }
      this._claimedChallenger = name;
      console.error(
        `[bridge] pad ${this.padIndex} challenger_claimed ${name} — later joiners are ` +
          'ignored until the next reset',
      );
      return;
    }
  }

  /**
   * Release the challenger slot so the next reset can arm a new challenger.
   * Called from the tail of handleReset and nowhere else.
   */
  _releaseChallengerSlot() {
    this._claimedChallenger = null;
    this._outOfPadReported.clear();
    this._challengerPresentLogged = null;
  }

  /**
   * Log the exhibition's status when — and only when — it CHANGES.
   *
   * The `state` message is `additionalProperties:false` on both validators and
   * the env blocks on exactly one `state` per `step`, so there is no wire slot
   * for "waiting for challenger": it is a bridge-log line by contract. Edge
   * triggered because handleStep runs five times a second.
   *
   * @param {object|null} handle The handle resolved for this window.
   */
  _logChallengerPresence(handle) {
    if (this._opponentIsBot()) {
      return;
    }
    const present = handle !== null;
    if (present === this._challengerPresentLogged) {
      return;
    }
    this._challengerPresentLogged = present;
    if (present) {
      console.error(
        `[bridge] pad ${this.padIndex} challenger_present ${handle.username} — match live`,
      );
      return;
    }
    const slot = this._challengerSlot();
    console.error(
      `[bridge] pad ${this.padIndex} challenger_absent ` +
        `${slot === null ? '(slot unclaimed)' : slot} — holding IDLE`,
    );
  }

  /**
   * The challenger's player entity in the learner's own view, or null.
   *
   * A PURE READ of the claimed slot (T3), by NAME. It claims nothing, so it is
   * safe to call from anywhere — including the scoreboard packet handler, via
   * _isChallengerName's live-resolve fallback, which is precisely where a
   * resolver that also claimed would let an incoming death decide who the
   * challenger was.
   *
   * Two nulls with different meanings, both handled the same by every caller:
   * the slot is free (nobody has walked into the pad yet), or the claimant is
   * not currently in view (they left, or died). The claim SURVIVES the second
   * case — the slot is theirs until a reset — so nobody inherits the match by
   * standing nearby, and the same person walking back in resumes it.
   *
   * @returns {object|null} A Mineflayer player entity, or null.
   */
  _resolveChallengerEntity() {
    const claimed = this._challengerSlot();
    if (claimed === null) {
      return null;
    }
    const entities = this._learnerEntities();
    if (entities === null) {
      return null;
    }
    // DEFENCE IN DEPTH: a claim can never be one of our own bots (_claimChallenger
    // excludes them), but `challengerUsername` is operator-supplied and a typo
    // naming the learner would otherwise point every behavioral read — and
    // ATTACK's target — at the agent itself.
    if (claimed === this.config.learnerUsername || claimed === this.config.dummyUsername) {
      return null;
    }
    for (const key of Object.keys(entities)) {
      const entity = entities[key];
      if (!entity || entity.type !== 'player') {
        continue;
      }
      if (entity.username === claimed) {
        return entity;
      }
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

  // -------------------------------------------------------------------------
  // HUMAN DEATH DETECTION (T2) — where `opponent_died` comes from when the
  // opponent has no connection of its own.
  //
  // WHY NOT HEALTH. Mineflayer never populates `entity.health` for anyone but
  // the connected bot. PR #32 repaired damage_dealt only by reading the dummy's
  // OWN connection; a human has none, so there is no health channel for them at
  // all. Nothing here may read `state.opponent.health` either: _snapshotOpponent
  // emits 0 for a human for want of a source, on a field labelled "PRIVILEGED
  // raw true health", so keying off it would report a kill on the first
  // observation of every exhibition match. _opponentHealth() returns null —
  // never 0 — for the same reason.
  //
  // WHY THE RAW PACKET FEED AND NOT MINEFLAYER'S SCOREBOARD PLUGIN. Reading
  // `bot.scoreboards[...]` or listening for `scoreUpdated` looks like the
  // obvious route and is DEAD on this server version. Both were checked at
  // primary source before this was written:
  //
  //   - node_modules/mineflayer/lib/plugins/scoreboard.js:41-46 gates the whole
  //     score path on `packet.action === 0`;
  //   - the 1.21.1 `scoreboard_score` packet HAS NO `action` FIELD — it is
  //     `{itemName, scoreName, value, display_name, number_format, styling}`
  //     (minecraft-data 1.21.1 protocol; the field was split out into a
  //     separate `reset_score` packet back in 1.20.3).
  //
  // So `packet.action` is `undefined`, the branch never runs, `scoreUpdated`
  // never fires, and `ScoreBoard.itemsMap` is never populated. Listening on
  // `bot._client` is therefore not a shortcut around the plugin — it is the
  // only working form of "read it from packets, not from chat". Do not
  // "simplify" this back to the plugin event; it would fail silently and take
  // human win detection with it. (`scoreboard_objective` DOES still carry
  // `action`, which is why the read-back below can use it.)
  // -------------------------------------------------------------------------

  /**
   * The learner's raw packet client, or null when there is none (every unit
   * fake, and any bot that has not connected yet).
   *
   * @returns {object|null}
   */
  _learnerPacketClient() {
    const client = this.learner ? this.learner._client : null;
    return client && typeof client.on === 'function' ? client : null;
  }

  /**
   * (Re-)wire the `rl_deaths` packet listeners. Idempotent: handlers are
   * removed from the client they were ADDED to before new ones are attached,
   * so a reconnect cannot double-register and count one death twice.
   *
   * Inert in 'bot' mode — no listeners at all. There the dummy's own `death`
   * event remains the single, unchanged source of opponent_died.
   */
  _wireDeathScoreboard() {
    const previous = this._deathScoreClient;
    if (previous && typeof previous.off === 'function') {
      if (this._boundOnScoreboardScore !== null) {
        previous.off('scoreboard_score', this._boundOnScoreboardScore);
      }
      if (this._boundOnResetScore !== null) {
        previous.off('reset_score', this._boundOnResetScore);
      }
      if (this._boundOnScoreboardObjective !== null) {
        previous.off('scoreboard_objective', this._boundOnScoreboardObjective);
      }
      if (this._boundOnScoreboardDisplay !== null) {
        previous.off('scoreboard_display_objective', this._boundOnScoreboardDisplay);
      }
    }
    this._deathScoreClient = null;
    this._boundOnScoreboardScore = null;
    this._boundOnResetScore = null;
    this._boundOnScoreboardObjective = null;
    this._boundOnScoreboardDisplay = null;

    if (this._opponentIsBot()) {
      return;
    }
    const client = this._learnerPacketClient();
    if (client === null) {
      return;
    }
    this._boundOnScoreboardScore = (packet) => this._onScoreboardScore(packet);
    this._boundOnResetScore = (packet) => this._onResetScore(packet);
    this._boundOnScoreboardObjective = (packet) => this._onScoreboardObjective(packet);
    this._boundOnScoreboardDisplay = (packet) => this._onScoreboardDisplay(packet);
    client.on('scoreboard_score', this._boundOnScoreboardScore);
    client.on('reset_score', this._boundOnResetScore);
    client.on('scoreboard_objective', this._boundOnScoreboardObjective);
    client.on('scoreboard_display_objective', this._boundOnScoreboardDisplay);
    this._deathScoreClient = client;
  }

  /**
   * Issue the two objective commands and VERIFY the server acted on them.
   *
   * A reply-less command proves nothing in this project — a `$`-macro can abort
   * the whole function and a `/fill` can no-op into unloaded chunks, both with
   * an empty log — so the objective is read BACK off the packet feed rather
   * than assumed. Confirmation is either packet the server can answer with:
   * `scoreboard_objective` (action 0) when the objective becomes tracked for
   * the first time, or `scoreboard_display_objective` when it was already
   * tracked from an earlier run and only the display is re-pinned.
   *
   * Never throws and never rejects: a scoreboard hiccup must not fail the whole
   * connect(). An unconfirmed objective is reported loudly and detection is
   * armed anyway — running unverified beats being silently switched off.
   *
   * @returns {Promise<boolean>} Whether the server echoed the objective back.
   */
  async _verifyDeathObjective() {
    if (this._opponentIsBot()) {
      return false;
    }
    // RE-ARM THE READ-BACK LATCH FIRST (S1). _deathObjectiveSeen is set by the
    // packet handlers and, without this line, never cleared: a second call
    // would see the PREVIOUS call's echo, return `confirmed: true` on its first
    // synchronous poll, and have verified nothing about the two commands it is
    // about to issue. connect() runs once today (bridge/run.js:353) so no
    // caller can reach that yet — T3's launcher and T5's reconnect story add
    // ones that can. A latch that can only be set is a read-back that has
    // quietly stopped reading back, which is this project's signature failure.
    this._deathObjectiveSeen = false;
    for (const command of formatDeathObjectiveCommands()) {
      this._sendCommand(this.learner, command);
    }
    let confirmed = false;
    try {
      confirmed = await waitForConfirmation(
        () => this._deathObjectiveSeen,
        this._deathObjectiveTimeoutMs,
        DEFAULT_READBACK.pollIntervalMs,
      );
    } catch (err) {
      confirmed = false;
    }
    if (confirmed) {
      // DRAIN THE BURST BEFORE ARMING (W2). The packet that confirms and the
      // score replay that follows it are ONE list: getStartTrackingPackets
      // builds [SetObjective, SetDisplayObjective, SetScore x N] and the server
      // writes the whole thing back to back. The poll above, though, runs on a
      // 50 ms timer against whatever the socket has already delivered. If the
      // list arrives in one synchronous emit batch the poll can only ever see
      // all of it; if it spans two TCP reads the poll can land in the gap and
      // arm detection while HISTORICAL scores are still coming in — each of
      // which then reads as an increase and is credited as a kill from a match
      // that ended before this process started. One further poll interval of
      // silence is what proves the burst is drained. It costs one interval on
      // the happy path and nothing on the timeout path, where the full budget
      // has already elapsed. `sleep` cannot reject, so this needs no guard.
      await sleep(DEFAULT_READBACK.pollIntervalMs);
    }
    // ARMED EITHER WAY, and deliberately so. Leaving detection disarmed on a
    // failed read-back would convert a loud, recoverable problem into the
    // silent one this whole path exists to avoid: an exhibition where the human
    // dies and the agent is never credited. What the flag actually buys is the
    // baseline — everything before this point is a replay of history, not news.
    this._deathObjectiveReady = true;
    if (!confirmed) {
      console.error(
        `[bridge] pad ${this.padIndex} ${RL_DEATHS_OBJECTIVE} objective NOT confirmed by the ` +
          `server within ${this._deathObjectiveTimeoutMs}ms — human win detection is armed but ` +
          'UNVERIFIED; check that the learner is opped and that ' +
          `"${formatDeathObjectiveCommands().join('" and "')}" were accepted`,
      );
    }
    return confirmed;
  }

  /**
   * A score changed on some holder's `rl_deaths` entry.
   *
   * @param {{itemName?:string, scoreName?:string, value?:number}} packet
   */
  _onScoreboardScore(packet) {
    if (!packet || packet.scoreName !== RL_DEATHS_OBJECTIVE) {
      return;
    }
    const name = typeof packet.itemName === 'string' ? packet.itemName : null;
    const value =
      typeof packet.value === 'number' && Number.isFinite(packet.value) ? packet.value : null;
    if (name === null || value === null) {
      // A malformed packet records nothing and must never take the bridge down.
      return;
    }
    this._recordDeathScore(name, value);
  }

  /**
   * A holder's score entry was REMOVED (1.20.3+ split this out of the score
   * packet; mineflayer's plugin handles it for neither). An absent entry reads
   * as 0, so this re-baselines rather than reporting anything.
   *
   * @param {{entity_name?:string, objective_name?:string}} packet
   */
  _onResetScore(packet) {
    if (!packet) {
      return;
    }
    const name = typeof packet.entity_name === 'string' ? packet.entity_name : null;
    if (name === null) {
      return;
    }
    // An omitted objective name means "every objective for this holder".
    const objective =
      typeof packet.objective_name === 'string' ? packet.objective_name : RL_DEATHS_OBJECTIVE;
    if (objective !== RL_DEATHS_OBJECTIVE) {
      return;
    }
    this._deathScores.set(name, 0);
  }

  /**
   * The objective was created (action 0) or removed (action 1) for this client.
   *
   * @param {{name?:string, action?:number}} packet
   */
  _onScoreboardObjective(packet) {
    if (!packet || packet.name !== RL_DEATHS_OBJECTIVE) {
      return;
    }
    if (packet.action === 0) {
      this._deathObjectiveSeen = true;
    } else if (packet.action === 1) {
      // The objective stopped being tracked: no further score packets can
      // arrive, so say so rather than silently reporting nothing forever.
      this._deathObjectiveSeen = false;
      console.error(
        `[bridge] pad ${this.padIndex} ${RL_DEATHS_OBJECTIVE} objective was removed server-side ` +
          '— human win detection has no source until it is re-added',
      );
    }
  }

  /**
   * The objective was (re-)pinned to a display slot. On a server that already
   * had it tracked from an earlier run this is the ONLY echo `setdisplay`
   * produces, so it counts as a read-back too.
   *
   * @param {{name?:string}} packet
   */
  _onScoreboardDisplay(packet) {
    if (packet && packet.name === RL_DEATHS_OBJECTIVE) {
      this._deathObjectiveSeen = true;
    }
  }

  /**
   * Fold one observed `rl_deaths` value into the baseline map, reporting an
   * INCREASE on the challenger's entry as opponent_died.
   *
   * A holder with no entry yet reads as 0, NOT as "unknown": `deathCount`
   * entries do not exist until the first death, so a challenger's first-ever
   * packet is `value: 1` at the moment they die — precisely the event AC3
   * exists for. Treating a first sighting as a baseline would swallow it.
   *
   * The replay that would otherwise be misread by that same rule (the burst
   * `setdisplay` triggers, which resends every pre-existing score) is handled
   * by _deathObjectiveReady, which is false for its whole duration.
   *
   * NOT gated by _suppressOpponentEvents — same deliberate choice as the dummy
   * `death` handler it stands in for (see wireDamageEvents). A death fired
   * during a reset window is discarded by the winning handleReset's
   * events.reset(); gating here would instead break detection mid-episode
   * whenever a concurrent reset happens to be in flight.
   *
   * @param {string} name The scoreboard holder (a username for players).
   * @param {number} value The holder's new `rl_deaths` value.
   */
  _recordDeathScore(name, value) {
    const prior = this._deathScores.has(name) ? this._deathScores.get(name) : 0;
    // The baseline advances unconditionally, including while priming and for
    // holders nobody is fighting, so the NEXT increment is measured correctly.
    this._deathScores.set(name, value);
    if (!this._deathObjectiveReady) {
      return;
    }
    if (value <= prior) {
      // No increase: a re-send of the same value, or an external re-baseline
      // (`/scoreboard players set|reset`). Neither is a death.
      return;
    }
    if (!this._isChallengerName(name)) {
      return;
    }
    this.events.recordOpponentDied();
  }

  /**
   * Whether a scoreboard holder name is the opponent this pad is fighting.
   *
   * `rl_deaths` is server-wide: the learner's own deaths, a neighbouring pad's
   * bots and any bystander all land on the same objective, so an unattributed
   * increment would credit the agent with somebody else's death.
   *
   * @param {string} name
   * @returns {boolean}
   */
  _isChallengerName(name) {
    // The next two guards are DEFENCE IN DEPTH and are deliberately not unit
    // tested — no test can reach them, which is the point. Nothing is wired in
    // 'bot' mode, and neither branch below can yield this pad's own usernames
    // (_resolveChallengerEntity excludes both, and the memory is only ever
    // written from a non-bot handle). They stay because every other layer is
    // one refactor from changing and the cost of being wrong here is not a
    // missed win but a FABRICATED one — the agent credited for its own death.
    if (this._opponentIsBot()) {
      return false;
    }
    if (name === this.config.learnerUsername || name === this.config.dummyUsername) {
      return false;
    }
    if (this.challengerUsername !== null) {
      return name === this.challengerUsername;
    }
    if (this._challengerDeathName !== null) {
      return name === this._challengerDeathName;
    }
    // DEFENCE IN DEPTH as well, and under the first-claimant latch no longer a
    // second attribution path. Reaching this line needs BOTH an unpinned
    // challengerUsername and an empty _challengerDeathName, and handleStep takes
    // the claim and writes that memory in ONE synchronous prologue
    // (_claimChallenger then _noteChallengerIdentity, no await between). So no
    // STEP can leave a claim outliving the memory, and with no reset in flight
    // this line is reached only with the slot null too: _challengerSlot() is
    // null, _opponentHandle() returns null, and the answer is false.
    //
    // A RESET can. handleReset clears the memory at its top and releases the
    // slot only at its tail, so the claim outlives the memory for as long as
    // that reset is in flight — and a THROWN gate propagates past the tail
    // without releasing, leaving the claim held for the retry. A score packet
    // landing in that interval reaches this line with the outgoing challenger
    // still claimed and can answer true; the opponent_died it records is
    // discarded by the winning handleReset's events.reset(), before that reset
    // acks (and where no reset ever wins, nothing is ever acked or drained).
    //
    // So the case this branch was written for — a death between the reset ack
    // and the episode's first step — is now DROPPED rather than attributed. A
    // few milliseconds, and dropping is the conservative direction: a missed
    // win, never a fabricated one. The branch stays because every layer above it
    // is one refactor from changing, and being wrong here costs a FABRICATED win.
    const handle = this._opponentHandle();
    return handle !== null && handle.isBot === false && handle.username === name;
  }

  /**
   * Remember WHO the opponent is, for attributing a later death packet.
   *
   * Called once per decision window with the handle handleStep already
   * resolved, so the memory and the window's swing always describe the same
   * person. No-op in 'bot' mode and when there is no challenger — a slot that
   * empties keeps the last name rather than blanking, because a player who
   * dies can drop out of the entity view in the same instant.
   *
   * @param {object|null} handle The handle resolved for this window.
   */
  _noteChallengerIdentity(handle) {
    if (handle !== null && handle.isBot === false && typeof handle.username === 'string') {
      this._challengerDeathName = handle.username;
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
  RL_DEATHS_OBJECTIVE,
  RL_DEATHS_DISPLAY_SLOT,
  PAD_INTERIOR_BOUNDS,
  // Pure, unit-testable logic.
  assertMacroInt,
  assertMacroUsername,
  isInsidePad,
  formatDeathObjectiveCommands,
  formatSetupPadCommand,
  formatResetPadCommand,
  formatHumanResetCommands,
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
