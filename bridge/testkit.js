// testkit.js — shared helpers for the `node --test` suites. TEST-ONLY: nothing
// in the production path (bot.js / actions.js / transport.js / run.js) requires
// this file, and it must stay that way.
//
// Deliberately NOT named `*.test.js` so the runner does not try to execute it as
// a test file (it declares no tests).
//
// ============================================================================
// MOCK FIDELITY — the rule this whole suite exists to enforce.
//   Mineflayer populates `health` ONLY on a bot's own connection
//   (lib/plugins/health.js: `bot.health = packet.health`, fed by the
//   update_health packet, which the server sends only about the receiving
//   client's own player). `prismarine-entity`'s Entity class defines NO health
//   field, so `entity.health` is `undefined` for every entity that is not the
//   bot itself. Health, hunger, XP and effects come only from a bot's own
//   connection; position, yaw, velocity and equipment are fine from the entity
//   view.
//
//   `damage_dealt` was identically zero for the entire life of this project
//   because its only recorder read `entity.health` off the learner's view of the
//   dummy — and the tests that "covered" it passed, because their fake entity
//   carried a `health` property reality does not have. A mock more capable than
//   reality tests nothing. Fakes must not carry a field mineflayer does not
//   populate.
// ============================================================================

'use strict';

/**
 * Wrap an ArenaBots' EventAggregator recorders with counting spies that delegate
 * to the real implementation.
 *
 * AC5 requires assertions on CALL COUNTS, not just final values, because a
 * double-registered handler and a correctly-registered one can produce identical
 * totals: on the health path the second copy sees a zero delta and records
 * nothing, so the total is right for the wrong reason. Call counts (and, on the
 * death path, which has no delta guard) are what actually distinguish them.
 *
 * Shared between bot.test.js and actions.test.js so a change to the recorder
 * surface cannot be applied to one file and silently missed in the other.
 *
 * @param {object} arena An ArenaBots instance; its `events` is patched in place.
 * @returns {{damageDealt:number[], damageTaken:number[], opponentDied:number, iDied:number}}
 *   Live call log: `damageDealt`/`damageTaken` collect each recorded amount in
 *   order; the death fields count invocations.
 */
function spyRecorders(arena) {
  const calls = { damageDealt: [], damageTaken: [], opponentDied: 0, iDied: 0 };
  const events = arena.events;
  const realDealt = events.recordDamageDealt.bind(events);
  const realTaken = events.recordDamageTaken.bind(events);
  const realOpponentDied = events.recordOpponentDied.bind(events);
  const realIDied = events.recordIDied.bind(events);

  events.recordDamageDealt = (amount) => {
    calls.damageDealt.push(amount);
    return realDealt(amount);
  };
  events.recordDamageTaken = (amount) => {
    calls.damageTaken.push(amount);
    return realTaken(amount);
  };
  events.recordOpponentDied = () => {
    calls.opponentDied += 1;
    return realOpponentDied();
  };
  events.recordIDied = () => {
    calls.iDied += 1;
    return realIDied();
  };
  return calls;
}

module.exports = { spyRecorders };
