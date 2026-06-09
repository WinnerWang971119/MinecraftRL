// transport.js — Raw TCP + newline-delimited JSON transport layer.
//
// Opens a TCP server (or client) socket, buffers partial reads across packet
// boundaries, emits complete JSON-lines messages, and serializes outgoing
// messages.  Provides a clean send(msg)/on('message', cb) interface so the
// rest of the bridge does not touch raw sockets.
//
// Owner: T7a (Environment/bridge track)
// TODO(T7a): implemented by task T7a
