import http from 'http';
import { WebSocketServer } from 'ws';
import express from 'express';
import * as Y from 'yjs';
import * as syncProtocol from 'y-protocols/sync';
import * as awarenessProtocol from 'y-protocols/awareness';
import * as encoding from 'lib0/encoding';
import * as decoding from 'lib0/decoding';

const PORT = process.env.PORT || 1234;
const HOST = process.env.HOST || 'localhost';

// ─── Room document store ──────────────────────────────────────────
// roomName -> { doc, awareness, connections: Map<ws, Set<clientId>> }
const rooms = new Map();

function getOrCreateRoom(roomName) {
  if (!rooms.has(roomName)) {
    const doc = new Y.Doc({ gc: true });
    const awareness = new awarenessProtocol.Awareness(doc);
    const connections = new Map(); // ws -> Set<clientId>

    // When the Y.Doc changes, broadcast the update to ALL peers in the room
    doc.on('update', (update, origin) => {
      const encoder = encoding.createEncoder();
      encoding.writeVarUint(encoder, 0); // messageSync
      syncProtocol.writeUpdate(encoder, update);
      const message = encoding.toUint8Array(encoder);

      connections.forEach((_, ws) => {
        if (ws.readyState === 1) {
          try { ws.send(message); } catch (_) {}
        }
      });
    });

    // When awareness changes, broadcast to all OTHER connections
    awareness.on('update', ({ added, updated, removed }, conn) => {
      const changed = added.concat(updated, removed);
      const encoder = encoding.createEncoder();
      encoding.writeVarUint(encoder, 1); // messageAwareness
      encoding.writeVarUint8Array(
        encoder,
        awarenessProtocol.encodeAwarenessUpdate(awareness, changed)
      );
      const message = encoding.toUint8Array(encoder);

      connections.forEach((_, ws) => {
        if (ws !== conn && ws.readyState === 1) {
          try { ws.send(message); } catch (_) {}
        }
      });
    });

    rooms.set(roomName, { doc, awareness, connections });
  }
  return rooms.get(roomName);
}

function cleanupRoom(roomName, ws) {
  const room = rooms.get(roomName);
  if (!room) return;

  room.connections.delete(ws);

  // Remove all awareness states associated with this connection
  const clientIds = room.awareness.getStates().keys();
  awarenessProtocol.removeAwarenessStates(
    room.awareness,
    Array.from(clientIds),
    null
  );

  // Destroy room if empty
  if (room.connections.size === 0) {
    room.doc.destroy();
    rooms.delete(roomName);
    console.log(`  🗑️  Room "${roomName}" destroyed (empty)`);
  }
}

// ─── HTTP server (Express + WebSocket) ────────────────────────────
const app = express();

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    rooms: Array.from(rooms.keys()).map((name) => ({
      name,
      connections: rooms.get(name).connections.size,
    })),
  });
});

const server = http.createServer(app);

// ─── WebSocket server ─────────────────────────────────────────────
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws, req) => {
  // Parse room name from URL path: ws://host:port/<roomName>
  const roomName = decodeURIComponent(req.url || '')
    .split('?')[0]
    .replace(/^\//, '') || 'default';

  const { doc, awareness } = getOrCreateRoom(roomName);
  console.log(`  ➕ Client joined room "${roomName}" (${rooms.get(roomName).connections.size + 1} peers)`);

  // ── Send sync step 1 to the new client ──
  const encoder = encoding.createEncoder();
  encoding.writeVarUint(encoder, 0); // messageSync
  syncProtocol.writeSyncStep1(encoder, doc);
  ws.send(encoding.toUint8Array(encoder));

  // ── Send current awareness states to the new client ──
  const awarenessStates = awareness.getStates();
  if (awarenessStates.size > 0) {
    const encAw = encoding.createEncoder();
    encoding.writeVarUint(encAw, 1); // messageAwareness
    encoding.writeVarUint8Array(
      encAw,
      awarenessProtocol.encodeAwarenessUpdate(
        awareness,
        Array.from(awarenessStates.keys())
      )
    );
    ws.send(encoding.toUint8Array(encAw));
  }

  // Track this connection
  const { connections } = rooms.get(roomName);
  connections.set(ws, new Set());

  // ── Handle incoming messages ──
  ws.on('message', (data) => {
    const decoder = decoding.createDecoder(new Uint8Array(data));
    const messageType = decoding.readVarUint(decoder);

    switch (messageType) {
      case 0: { // sync
        const replyEncoder = encoding.createEncoder();
        encoding.writeVarUint(replyEncoder, 0);
        syncProtocol.readSyncMessage(decoder, replyEncoder, doc, ws);
        if (encoding.length(replyEncoder) > 1) {
          ws.send(encoding.toUint8Array(replyEncoder));
        }
        break;
      }
      case 1: { // awareness
        awarenessProtocol.applyAwarenessUpdate(
          awareness,
          decoding.readVarUint8Array(decoder),
          ws
        );
        break;
      }
      case 2: { // auth — not used in this simple version
        break;
      }
    }
  });

  // ── Handle close ──
  ws.on('close', () => {
    console.log(`  ➖ Client left room "${roomName}"`);
    cleanupRoom(roomName, ws);
  });

  // ── Handle error ──
  ws.on('error', (err) => {
    console.error(`  ⚠️  WebSocket error in "${roomName}":`, err.message);
    cleanupRoom(roomName, ws);
  });
});

// ── Upgrade HTTP → WebSocket ──
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});

// ── Start ──
server.listen(PORT, HOST, () => {
  console.log(`
╭──────────────────────────────────────────╮
│  📝 Collaborative Editor Server          │
│  ───────────────────────                 │
│  HTTP  : http://${HOST}:${PORT}/         │
│  WS    : ws://${HOST}:${PORT}/<room>     │
│  Health: http://${HOST}:${PORT}/health   │
╰──────────────────────────────────────────╯
  `);
});
