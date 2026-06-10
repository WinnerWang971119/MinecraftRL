// transport.js — Raw TCP + newline-delimited JSON transport layer (T7a).
//
// One localhost TCP server, ONE connection per arena (the frozen wire contract
// has no arena id — multi-arena = multiple independent connections). The
// transport buffers partial reads across TCP packet boundaries, frames
// newline-delimited JSON, emits parsed inbound messages, and serializes
// outbound objects as `JSON.stringify(obj) + "\n"`. The rest of the bridge
// never touches a raw socket.
//
// The wire contract is FROZEN in bridge/schema.json / bridge/schema.md /
// bridge/messages.py. This file matches it exactly:
//   Python -> Node : reset{episode,seed}, step{action 0..7}, close
//   Node   -> Python: state{self,opponent,events,arena,tick,code_version},
//                     reset_ack{ok,readback}
// The Python client (env/mc_pvp_env.py: TcpBridgeClient) connects to
// 127.0.0.1:5555 by default, sends compact JSON lines, tolerates blank
// keep-alive lines, and buffers partial reads on its side too.
//
// ============================================================================
// VERIFIED HERE (node --test, NO live server):
//   - LineFramer reassembles messages split byte-by-byte across chunks.
//   - LineFramer handles several whole messages in one chunk.
//   - LineFramer skips blank keep-alive lines.
//   - encodeMessage(...) round-trips through the framer (compact, +"\n").
//   - Outbound state / reset_ack are validated against the schema fields;
//     malformed outbound throws loudly.
// LIVE-ONLY (requires the Paper 1.21.1 server, per server/compat_check.md):
//   - TC10  reset -> step -> state round-trip over a real socket + bots.
//   - TC14  reset determinism (same seed -> same readback).
//   These need the live Mineflayer handshake and are a documented human
//   follow-up (server/compat_check.md "Live handshake").
// ============================================================================
//
// Owner: T7a (Environment/bridge track)

'use strict';

const net = require('node:net');
const { EventEmitter } = require('node:events');

// ---------------------------------------------------------------------------
// Frozen wire constants. Mirror bridge/schema.json + bridge/messages.py.
// ---------------------------------------------------------------------------

/** Default loopback bind host. Matches env TcpBridgeClient host "127.0.0.1". */
const DEFAULT_HOST = '127.0.0.1';

/** Default bridge TCP port. Matches env TcpBridgeClient port 5555. */
const DEFAULT_PORT = 5555;

/** The single frame delimiter: one UTF-8 newline. No "\n" appears inside a message. */
const NEWLINE = '\n';

/** Inclusive bounds of the discrete action index (step.action). FROZEN: 0..7. */
const ACTION_MIN = 0;
const ACTION_MAX = 7;

/** Inbound (Python -> Node) message types. */
const INBOUND_TYPES = Object.freeze(['reset', 'step', 'close']);

/** Outbound (Node -> Python) message types. */
const OUTBOUND_TYPES = Object.freeze(['state', 'reset_ack']);

// ---------------------------------------------------------------------------
// LineFramer — the pure, testable JSON-lines decoder.
//
// TCP is a byte stream: a single chunk may carry half a message, several
// messages, or a message split across packets. The framer ACCUMULATES bytes
// and only parses up to each "\n". It is deliberately free of any socket so
// `node --test` can feed it fragmented Buffers / strings and assert exact
// message reassembly. It mirrors the Python recv() loop in env/mc_pvp_env.py.
// ---------------------------------------------------------------------------

class LineFramer {
  constructor() {
    // Accumulated bytes received but not yet framed into a complete line.
    // Persisted across push() calls so a message split across packets is
    // reassembled correctly. Kept as a Buffer so multi-byte UTF-8 sequences
    // split across chunk boundaries are never decoded mid-character.
    this._buf = Buffer.alloc(0);
  }

  /**
   * Feed one chunk of received bytes and return every complete message that is
   * now available. A chunk may yield zero, one, or many messages; an incomplete
   * trailing line is retained in the internal buffer for the next push().
   *
   * Blank lines (empty or whitespace-only, the keep-alive convention the Python
   * side tolerates) are skipped, not parsed.
   *
   * @param {Buffer|Uint8Array|string} chunk Raw bytes (or a UTF-8 string).
   * @returns {Array<object>} Parsed JSON objects, in arrival order.
   * @throws {Error} If a complete line is not valid JSON or not a JSON object.
   */
  push(chunk) {
    const incoming = Buffer.isBuffer(chunk)
      ? chunk
      : Buffer.from(chunk, 'utf8');
    this._buf = this._buf.length === 0
      ? incoming
      : Buffer.concat([this._buf, incoming]);

    const messages = [];
    let newlineIndex;
    // Drain every complete "\n"-terminated line currently in the buffer.
    while ((newlineIndex = this._buf.indexOf(0x0a)) >= 0) {
      const rawLine = this._buf.subarray(0, newlineIndex);
      // Drop the line AND its trailing newline from the buffer.
      this._buf = this._buf.subarray(newlineIndex + 1);

      const text = rawLine.toString('utf8').trim();
      if (text === '') {
        // Tolerate blank keep-alive lines: skip and keep draining.
        continue;
      }

      let decoded;
      try {
        decoded = JSON.parse(text);
      } catch (err) {
        throw new Error(`bridge received an invalid JSON line: ${err.message}`);
      }
      if (decoded === null || typeof decoded !== 'object' || Array.isArray(decoded)) {
        throw new Error(
          `a message line must decode to an object, got ${describeJsonType(decoded)}`,
        );
      }
      messages.push(decoded);
    }
    return messages;
  }

  /** Number of bytes currently buffered (an unterminated partial line). */
  get pendingBytes() {
    return this._buf.length;
  }

  /** Drop any buffered partial line (e.g. on disconnect). */
  reset() {
    this._buf = Buffer.alloc(0);
  }
}

/** Human-readable JSON type name for error messages. */
function describeJsonType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

// ---------------------------------------------------------------------------
// Outbound validation + encoding.
//
// Validate/shape outbound `state` / `reset_ack` against the schema fields and
// surface errors loudly (a malformed outbound message is a bridge bug, not a
// recoverable condition). This mirrors the relevant Node->Python branches of
// bridge/schema.json and the validator in bridge/messages.py. Inbound messages
// are validated by the Python side (messages.validate); the Node side only
// needs to produce conformant outbound frames.
// ---------------------------------------------------------------------------

class WireError extends Error {
  constructor(message) {
    super(message);
    this.name = 'WireError';
  }
}

/** True for a finite JSON number (excludes NaN/Infinity, which JSON cannot encode). */
function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isBoolean(value) {
  return typeof value === 'boolean';
}

function isString(value) {
  return typeof value === 'string';
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireField(cond, message) {
  if (!cond) {
    throw new WireError(message);
  }
}

/** Validate a [x,y,z] world-frame triple of finite numbers. */
function validateVec3(value, where) {
  requireField(Array.isArray(value), `${where} must be a [x,y,z] array`);
  requireField(value.length === 3, `${where} must have exactly 3 elements, got ${value.length}`);
  for (let i = 0; i < 3; i += 1) {
    requireField(isFiniteNumber(value[i]), `${where}[${i}] must be a finite number`);
  }
}

/** Require exactly `keys` on `obj` (no missing, no extras) — mirrors additionalProperties:false. */
function requireExactKeys(obj, keys, where) {
  requireField(isPlainObject(obj), `${where} must be an object`);
  const present = Object.keys(obj);
  const allowed = new Set(keys);
  for (const k of keys) {
    requireField(Object.prototype.hasOwnProperty.call(obj, k), `${where} missing required field "${k}"`);
  }
  for (const k of present) {
    requireField(allowed.has(k), `${where} has unexpected field "${k}"`);
  }
}

function validateSelf(self) {
  requireExactKeys(
    self,
    ['pos', 'yaw', 'pitch', 'velocity', 'on_ground', 'health', 'held_item', 'attack_cooldown'],
    'state.self',
  );
  validateVec3(self.pos, 'state.self.pos');
  requireField(isFiniteNumber(self.yaw), 'state.self.yaw must be a finite number');
  requireField(isFiniteNumber(self.pitch), 'state.self.pitch must be a finite number');
  validateVec3(self.velocity, 'state.self.velocity');
  requireField(isBoolean(self.on_ground), 'state.self.on_ground must be a boolean');
  requireField(isFiniteNumber(self.health), 'state.self.health must be a finite number');
  requireField(isString(self.held_item), 'state.self.held_item must be a string');
  requireField(isFiniteNumber(self.attack_cooldown), 'state.self.attack_cooldown must be a finite number');
}

function validateOpponent(opp) {
  requireExactKeys(opp, ['pos', 'yaw', 'pitch', 'velocity', 'health'], 'state.opponent');
  validateVec3(opp.pos, 'state.opponent.pos');
  requireField(isFiniteNumber(opp.yaw), 'state.opponent.yaw must be a finite number');
  requireField(isFiniteNumber(opp.pitch), 'state.opponent.pitch must be a finite number');
  validateVec3(opp.velocity, 'state.opponent.velocity');
  // PRIVILEGED: opponent.health is RAW true health. It is on the wire (reward
  // may read it) but MUST NEVER reach the observation (gated by the
  // PerceptionFilter, T12). The transport only checks it is a number.
  requireField(isFiniteNumber(opp.health), 'state.opponent.health must be a finite number');
}

function validateEvents(events) {
  requireExactKeys(events, ['damage_dealt', 'damage_taken', 'i_died', 'opponent_died'], 'state.events');
  requireField(isFiniteNumber(events.damage_dealt), 'state.events.damage_dealt must be a finite number');
  requireField(events.damage_dealt >= 0, 'state.events.damage_dealt must be >= 0');
  requireField(isFiniteNumber(events.damage_taken), 'state.events.damage_taken must be a finite number');
  requireField(events.damage_taken >= 0, 'state.events.damage_taken must be >= 0');
  requireField(isBoolean(events.i_died), 'state.events.i_died must be a boolean');
  requireField(isBoolean(events.opponent_died), 'state.events.opponent_died must be a boolean');
}

function validateArena(arena) {
  requireExactKeys(arena, ['wall_distances'], 'state.arena');
  const wd = arena.wall_distances;
  requireField(Array.isArray(wd), 'state.arena.wall_distances must be an array');
  for (let i = 0; i < wd.length; i += 1) {
    requireField(isFiniteNumber(wd[i]), `state.arena.wall_distances[${i}] must be a finite number`);
  }
}

function validateState(msg) {
  requireExactKeys(
    msg,
    ['type', 'self', 'opponent', 'events', 'arena', 'tick', 'code_version'],
    'state',
  );
  validateSelf(msg.self);
  validateOpponent(msg.opponent);
  validateEvents(msg.events);
  validateArena(msg.arena);
  requireField(Number.isInteger(msg.tick), 'state.tick must be an integer');
  requireField(msg.tick >= 0, 'state.tick must be >= 0');
  requireField(isString(msg.code_version), 'state.code_version must be a string');
}

function validateResetAck(msg) {
  requireExactKeys(msg, ['type', 'ok', 'readback'], 'reset_ack');
  requireField(isBoolean(msg.ok), 'reset_ack.ok must be a boolean');
  requireField(isPlainObject(msg.readback), 'reset_ack.readback must be an object');
}

const OUTBOUND_VALIDATORS = Object.freeze({
  state: validateState,
  reset_ack: validateResetAck,
});

/**
 * Validate an outbound (Node -> Python) message against the frozen schema.
 * Throws a WireError on any violation. Returns the message unchanged on success.
 *
 * @param {object} msg An outbound message object (must carry a `type`).
 * @returns {object} The same `msg`.
 * @throws {WireError} If `msg` is not a valid outbound message.
 */
function validateOutbound(msg) {
  requireField(isPlainObject(msg), 'outbound message must be an object');
  requireField(isString(msg.type), "outbound message missing string 'type' discriminator");
  const validator = OUTBOUND_VALIDATORS[msg.type];
  requireField(
    validator !== undefined,
    `unknown outbound message type "${msg.type}"; expected one of ${OUTBOUND_TYPES.join(', ')}`,
  );
  validator(msg);
  return msg;
}

/**
 * Encode an outbound message to a single newline-terminated JSON line.
 *
 * Validates against the frozen schema first (loud failure on malformed output),
 * then emits compact JSON (no spaces) + a single trailing "\n" — exactly what
 * the Python framer in env/mc_pvp_env.py splits on. JSON.stringify naturally
 * omits `undefined` values, but the validator has already rejected those.
 *
 * @param {object} msg An outbound message object.
 * @returns {string} The encoded `JSON.stringify(msg) + "\n"`.
 * @throws {WireError} If `msg` is not a valid outbound message.
 */
function encodeMessage(msg) {
  validateOutbound(msg);
  return JSON.stringify(msg) + NEWLINE;
}

// ---------------------------------------------------------------------------
// BridgeServer — the localhost TCP JSON-lines server (one connection per arena).
//
// Listens on loopback; accepts ONE active connection at a time (one arena per
// process for kickoff — multi-arena is the deferred arena.js). Each inbound
// complete line is parsed by a per-connection LineFramer and emitted as a
// 'message' event. `send(obj)` validates + encodes + writes one frame.
//
// Events:
//   'listening' (address)        server bound and accepting
//   'connection'                 a client (the Python env) connected
//   'message'   (msg)            one parsed inbound message
//   'disconnect'                 the active client connection ended
//   'error'     (err)            socket / framing / server error (loud)
// ---------------------------------------------------------------------------

class BridgeServer extends EventEmitter {
  /**
   * @param {object} [options]
   * @param {string} [options.host=DEFAULT_HOST] Loopback bind host.
   * @param {number} [options.port=DEFAULT_PORT] TCP port (matches the Python client).
   */
  constructor(options = {}) {
    super();
    this.host = options.host !== undefined ? options.host : DEFAULT_HOST;
    this.port = options.port !== undefined ? options.port : DEFAULT_PORT;

    /** @type {net.Server|null} */
    this._server = null;
    /** @type {net.Socket|null} The single active arena connection. */
    this._socket = null;
    /** @type {LineFramer} Per-connection inbound framer. */
    this._framer = new LineFramer();
  }

  /** True while a client connection is active. */
  get isConnected() {
    return this._socket !== null && !this._socket.destroyed;
  }

  /**
   * Bind the loopback TCP server and start accepting the single arena
   * connection. Resolves once the server is listening.
   *
   * @returns {Promise<{address: string, port: number}>}
   */
  listen() {
    if (this._server !== null) {
      return Promise.reject(new Error('BridgeServer is already listening'));
    }
    return new Promise((resolve, reject) => {
      const server = net.createServer((socket) => this._onConnection(socket));
      this._server = server;

      const onError = (err) => {
        // Bind-time failures (e.g. EADDRINUSE) reject the listen() promise.
        this._server = null;
        reject(err);
      };
      server.once('error', onError);

      server.listen(this.port, this.host, () => {
        server.removeListener('error', onError);
        // After bind, surface server-level errors through the event channel.
        server.on('error', (err) => this.emit('error', err));
        const addr = server.address();
        const info = { address: addr.address, port: addr.port };
        this.emit('listening', info);
        resolve(info);
      });
    });
  }

  /**
   * Wire a newly accepted client socket. If an old connection is still
   * registered, ADOPT the new one and drop the stale socket: the env client is
   * the only legitimate peer (one logical connection per arena process), and a
   * new connect means it abandoned the old socket — typically its documented
   * single-reconnect recovery racing our 'close' event on loopback. The
   * previous behavior (refuse the newcomer via socket.destroy(err)) emitted
   * 'error' on a socket with no listeners attached yet, which is process-fatal
   * and killed the bridge during the first live run. A genuinely concurrent
   * second client now steals the stream; that is operator error, made visible
   * by the 'disconnect'/'connection' events the launcher logs.
   */
  _onConnection(socket) {
    if (this._socket !== null && !this._socket.destroyed) {
      const stale = this._socket;
      // Detach before destroying so the stale socket's 'close' handler cannot
      // clobber the newly adopted socket.
      this._socket = null;
      stale.destroy();
    }

    this._socket = socket;
    this._framer.reset();
    // Disable Nagle: the bridge is request/response per decision step, so send
    // small command/state lines immediately rather than coalescing them.
    socket.setNoDelay(true);

    socket.on('data', (chunk) => this._onData(chunk));
    socket.on('error', (err) => this.emit('error', err));
    socket.once('close', () => {
      if (this._socket === socket) {
        this._socket = null;
        this._framer.reset();
      }
      this.emit('disconnect');
    });

    this.emit('connection');
  }

  /** Frame inbound bytes and emit each complete message. Framing errors are loud. */
  _onData(chunk) {
    let messages;
    try {
      messages = this._framer.push(chunk);
    } catch (err) {
      // A malformed line is unrecoverable for this stream: surface it and drop
      // the connection so the Python side reconnects from a clean state.
      this.emit('error', err);
      if (this._socket !== null) {
        this._socket.destroy(err);
      }
      return;
    }
    for (const msg of messages) {
      this.emit('message', msg);
    }
  }

  /**
   * Validate, encode, and write one outbound message to the active connection.
   *
   * @param {object} msg An outbound `state` or `reset_ack` object.
   * @returns {boolean} The socket write() backpressure result.
   * @throws {WireError} If `msg` is malformed (loud — a bridge bug).
   * @throws {Error} If there is no active connection.
   */
  send(msg) {
    const line = encodeMessage(msg);
    if (this._socket === null || this._socket.destroyed) {
      throw new Error('cannot send: no active bridge connection');
    }
    return this._socket.write(line);
  }

  /**
   * Drop the active client connection (if any) WITHOUT stopping the server.
   * This is the per-episode `close` semantics: the env opens a fresh client
   * per episode and sends `close` when it is done with this one, so only the
   * client socket goes away — the listener (and the bots above us) stay up
   * for the next episode's connection. Idempotent.
   */
  dropConnection() {
    const socket = this._socket;
    this._socket = null;
    this._framer.reset();
    if (socket !== null && !socket.destroyed) {
      socket.destroy();
    }
  }

  /**
   * Close the active connection and stop the server. Idempotent.
   * @returns {Promise<void>}
   */
  close() {
    return new Promise((resolve) => {
      const socket = this._socket;
      this._socket = null;
      this._framer.reset();
      if (socket !== null && !socket.destroyed) {
        socket.destroy();
      }
      const server = this._server;
      this._server = null;
      if (server === null) {
        resolve();
        return;
      }
      server.close(() => resolve());
    });
  }
}

module.exports = {
  // Constants (mirror the frozen contract).
  DEFAULT_HOST,
  DEFAULT_PORT,
  NEWLINE,
  ACTION_MIN,
  ACTION_MAX,
  INBOUND_TYPES,
  OUTBOUND_TYPES,
  // Pure framing + encoding (unit-testable without a socket).
  LineFramer,
  encodeMessage,
  validateOutbound,
  WireError,
  // TCP server (one connection per arena).
  BridgeServer,
};
