// run.js — process entry for the bridge (RUNBOOK Step 0).
//
// Does the four-call wiring that takes ArenaBots live: construct, route
// inbound reset/step/close to the handlers, connect both bots, then open the
// TCP port for the Python env. Start Paper first, then `npm start`, then the
// Python driver.
//
// Two deliberate choices beyond the minimal wiring:
//   - Bots connect BEFORE listen(): the port only opens once the bridge can
//     actually serve a reset. If Paper is down, createBot fails and the
//     process exits 1 instead of opening a port that leads nowhere.
//   - 'error' listeners everywhere: BridgeServer and the Mineflayer bots are
//     EventEmitters, and an 'error' event with no listener crashes the
//     process. The M1 bar is zero crashes over >=100 episodes, so socket and
//     bot errors are logged and the Python side reconnects/retries.
//
// Owner: RUNBOOK Step 0 (go-live wiring)

'use strict';

const { ArenaBots } = require('./bot');

async function main() {
  const bots = new ArenaBots(); // BridgeServer on 127.0.0.1:5555

  bots.transport.on('error', (err) => console.error('[bridge] transport error:', err));
  bots.transport.on('connection', () => console.error('[bridge] env connected'));
  bots.transport.on('disconnect', () => console.error('[bridge] env disconnected'));

  bots.wireTransport(); // route reset/step/close -> handlers

  await bots.connect(); // spawn learner_bot + dummy_bot (requires Paper up + opped)

  for (const [name, bot] of [['learner', bots.learner], ['dummy', bots.dummy]]) {
    bot.on('error', (err) => console.error(`[bridge] ${name} bot error:`, err));
    bot.on('kicked', (reason) => console.error(`[bridge] ${name} bot kicked:`, reason));
    bot.on('end', (reason) => console.error(`[bridge] ${name} bot disconnected:`, reason));
  }

  const { address, port } = await bots.transport.listen();
  console.error(`[bridge] listening on ${address}:${port}, both bots spawned`);

  process.once('SIGINT', () => {
    console.error('[bridge] SIGINT, shutting down');
    bots.close().then(() => process.exit(0));
  });
}

main().catch((err) => {
  console.error('[bridge] fatal:', err);
  process.exit(1);
});
