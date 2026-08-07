#!/usr/bin/env node
// probe_world.js — verify what the arena world is actually made of (task T7).
//
// This is the script that produced the block-composition result recorded in
// server/compat_check.md ("Block composition below the arena — AC17"). It is
// committed so that result is reproducible rather than a claim: T5's reading of
// wander-offs as STRANDINGS rather than DEATHS rests entirely on it.
//
// It answers three questions against a live server, with no human at a keyboard:
//   1. What blocks exist in a full-height column, y=319 down to y=-64 — inside
//      the arena pad and outside its footprint (where an edge-walker lands)?
//   2. What are the damage-relevant gamerules actually set to right now?
//   3. Do both bots join, and are they opped?
//
// It is a STANDALONE mineflayer client. It does NOT speak to the Node bridge and
// does not touch the bridge's TCP port, so it cannot disturb a training run's
// single-client BridgeServer. It only needs the mineflayer already installed in
// bridge/node_modules.
//
// Usage (with the Paper server already running):
//     node server/tools/probe_world.js
//     node server/tools/probe_world.js --host 127.0.0.1 --port 25565
//     node server/tools/probe_world.js --pad-x 512 --pad-z 0   # a non-zero pad
//
// Exit codes: 0 = probe completed, 1 = probe could not run (join failed, etc.).

const path = require('path');

// mineflayer lives in bridge/node_modules; this tool adds no dependency of its own.
const BRIDGE_MODULES = path.resolve(__dirname, '..', '..', 'bridge', 'node_modules');
let mineflayer;
let Vec3;
try {
  mineflayer = require(path.join(BRIDGE_MODULES, 'mineflayer'));
  ({ Vec3 } = require(path.join(BRIDGE_MODULES, 'vec3')));
} catch (err) {
  console.error('Could not load mineflayer from bridge/node_modules.');
  console.error('Run `npm ci` in bridge/ first. Original error:', err.message);
  process.exit(1);
}

// --- argv ------------------------------------------------------------------
function parseArgs(argv) {
  const opts = {
    host: '127.0.0.1',
    port: 25565,
    version: '1.21.1',
    learner: 'learner_bot',
    dummy: 'dummy_bot',
    padX: 0,
    padZ: 0,
  };
  const numeric = new Set(['port', 'padX', 'padZ']);
  const alias = {
    '--host': 'host', '--port': 'port', '--version': 'version',
    '--learner': 'learner', '--dummy': 'dummy',
    '--pad-x': 'padX', '--pad-z': 'padZ',
  };
  for (let i = 2; i < argv.length; i += 2) {
    const key = alias[argv[i]];
    if (!key) {
      console.error(`Unknown argument: ${argv[i]}`);
      process.exit(1);
    }
    const raw = argv[i + 1];
    if (raw === undefined) {
      console.error(`Missing value for ${argv[i]}`);
      process.exit(1);
    }
    if (numeric.has(key)) {
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        console.error(`${argv[i]} expects a number, got: ${raw}`);
        process.exit(1);
      }
      opts[key] = n;
    } else {
      opts[key] = raw;
    }
  }
  return opts;
}

const opts = parseArgs(process.argv);

// World height for 1.21 overworld.
const WORLD_TOP = 319;
const WORLD_BOTTOM = -64;
const AIR = new Set(['air', 'cave_air', 'void_air']);

// Pad footprint (arena setup): floor y=63 spanning x = anchor-8 .. anchor+16,
// z = anchor-12 .. anchor+12. The "outside" columns are past those edges — the
// ground an agent that walks off the platform actually lands on.
const COLUMNS = [
  { dx: 0, dz: 0, label: 'INSIDE pad footprint (learner spawn column)' },
  { dx: 20, dz: 0, label: 'OUTSIDE footprint, +X (past the x=+16 edge)' },
  { dx: 0, dz: 20, label: 'OUTSIDE footprint, +Z (past the z=+12 edge)' },
];

const GAMERULES = ['naturalRegeneration', 'fallDamage', 'drowningDamage', 'fireDamage', 'freezeDamage'];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function connect(username) {
  return new Promise((resolve, reject) => {
    const bot = mineflayer.createBot({
      host: opts.host, port: opts.port, username,
      version: opts.version, auth: 'offline',
    });
    const cleanup = () => {
      bot.removeListener('spawn', onSpawn);
      bot.removeListener('error', onErr);
      bot.removeListener('kicked', onKick);
    };
    const onSpawn = () => { cleanup(); resolve(bot); };
    const onErr = (e) => { cleanup(); reject(e); };
    const onKick = (r) => { cleanup(); reject(new Error(`kicked: ${JSON.stringify(r)}`)); };
    bot.once('spawn', onSpawn);
    bot.once('error', onErr);
    bot.once('kicked', onKick);
  });
}

// Send a command as the bot and collect whatever the server says back.
// A non-opped bot gets a permission/unknown-command reply, which is exactly how
// this doubles as an op check.
function runCommand(bot, cmd, ms = 1500) {
  return new Promise((resolve) => {
    const lines = [];
    const onMsg = (jsonMsg) => lines.push(jsonMsg.toString());
    bot.on('message', onMsg);
    bot.chat(cmd);
    setTimeout(() => { bot.removeListener('message', onMsg); resolve(lines); }, ms);
  });
}

function scanColumn(bot, x, z) {
  const found = [];
  let unloaded = 0;
  for (let y = WORLD_TOP; y >= WORLD_BOTTOM; y--) {
    const block = bot.blockAt(new Vec3(x, y, z));
    if (block === null) { unloaded++; continue; }
    if (!AIR.has(block.name)) found.push({ y, name: `minecraft:${block.name}` });
  }
  return { found, unloaded, total: WORLD_TOP - WORLD_BOTTOM + 1 };
}

async function main() {
  console.log('=== arena world probe ===');
  console.log(`server ${opts.host}:${opts.port}  mc ${opts.version}  pad anchor (${opts.padX}, ${opts.padZ})`);

  const t0 = Date.now();
  const learner = await connect(opts.learner);
  console.log(`[join] ${opts.learner} spawned at ${learner.entity.position} (+${Date.now() - t0}ms)`);

  // Deliberately immediate: two logins from one IP back-to-back also exercises
  // bukkit.yml's connection-throttle (see compat_check.md).
  const t1 = Date.now();
  const dummy = await connect(opts.dummy);
  console.log(`[join] ${opts.dummy} spawned at ${dummy.entity.position} (+${Date.now() - t1}ms later)`);
  console.log('[join] both bots joined without a throttle kick.');

  await sleep(3000); // let chunks stream in

  console.log('\n=== gamerules + op check (commands issued as the learner) ===');
  console.log('    (a NON-opped bot cannot see these commands at all, so a real');
  console.log('     value coming back is itself proof the bot is opped)');
  for (const rule of GAMERULES) {
    const out = await runCommand(learner, `/gamerule ${rule}`);
    console.log(`  /gamerule ${rule}  ->  ${JSON.stringify(out)}`);
  }
  for (const bot of [learner, dummy]) {
    const out = await runCommand(bot, '/gamerule naturalRegeneration');
    const opped = out.some((l) => l.includes('is currently set to'));
    console.log(`  op check ${bot.username.padEnd(14)} -> ${opped ? 'OPPED' : `NOT OPPED (${JSON.stringify(out)})`}`);
  }

  console.log(`\n=== BLOCK COLUMN SCAN (y=${WORLD_TOP} down to y=${WORLD_BOTTOM}) ===`);
  for (const col of COLUMNS) {
    const x = opts.padX + col.dx;
    const z = opts.padZ + col.dz;
    const { found, unloaded, total } = scanColumn(learner, x, z);
    console.log(`\n--- column x=${x}, z=${z} : ${col.label} ---`);
    console.log(`    (${unloaded} of ${total} y-levels returned null / unloaded)`);
    if (found.length === 0) {
      console.log('    NO NON-AIR BLOCKS ANYWHERE IN THIS COLUMN (true void)');
    } else {
      for (const b of found) console.log(`    y=${String(b.y).padStart(4)}  ${b.name}`);
    }
  }

  console.log('\n=== interpretation ===');
  console.log('Solid ground in the OUTSIDE columns means an agent that walks off the');
  console.log('pad LANDS ALIVE and is stranded (given fallDamage false) — a timeout,');
  console.log('never a death. No non-air blocks there would mean reachable void.');

  learner.quit();
  dummy.quit();
  await sleep(1000);
  process.exit(0);
}

main().catch((err) => {
  console.error('PROBE FAILED:', err.message || err);
  console.error('Is the Paper server running? Start it with server/setup/start.sh');
  process.exit(1);
});
