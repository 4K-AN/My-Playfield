# 📝 Collaborative Text Editor with CRDTs

A real-time collaborative text editor built with **React**, **CodeMirror 6**, **Yjs** (CRDT library), and **WebSockets**. Multiple users can edit the same document simultaneously — conflicts are resolved automatically without a central server.

---

## ✨ Features

- **Real-time sync** — edits appear on all connected clients in milliseconds
- **Conflict-free merging** — CRDTs (Yjs) resolve concurrent edits deterministically
- **Presence indicators** — see who's online and where their cursor is
- **Offline support** — keep editing during disconnects; changes merge on reconnect
- **Undo / Redo** — CRDT-aware undo manager
- **Named rooms** — use `?room=my-room` in the URL to create separate documents
- **Dark theme** — easy on the eyes for long editing sessions

---

## 🧠 CRDTs — Conflict-Free Replicated Data Types

### What is a CRDT?

A **Conflict-free Replicated Data Type** (CRDT) is a data structure that lets multiple users edit the same document simultaneously **without requiring a central lock or conflict resolution server**. Each peer maintains an independent copy, and the data structure is designed so that **all copies automatically converge to the same state** — even if edits happened while disconnected.

There are two main approaches to CRDTs:

1. **State-based (CvRDT)** — peers send their full state to each other; states merge via a monotonic join function.
2. **Operation-based (CmRDT)** — peers broadcast operations; operations are commutative so order doesn't matter.

Yjs implements a **state-based CRDT with operational deltas**, using a technique called **Internal Re-structuring** built on **Lamport timestamps**.

### How Yjs Resolves Conflicts

#### 1. Unique Identifiers

Every operation in Yjs is tagged with a **globally unique identifier**:

```
(clientID, clock)
```

- **`clientID`** — a random 32-bit integer assigned when a Y.Doc is created
- **`clock`** — a **Lamport clock** (logical timestamp) that increments with each local operation

This pair `(clientID, clock)` is unique across all peers in the system.

#### 2. Insertion Strategy — Split & Merge

Yjs stores document content in a data structure conceptually similar to a **linked list of fragments**. Each fragment holds a contiguous run of characters from the same origin.

When two users insert at the same logical position:

```
User A inserts "X" at position 5   →   (clientID_A, clock=10)
User B inserts "Y" at position 5   →   (clientID_B, clock=7)
```

Yjs **splits** the existing fragment at that position and inserts both new items. The final order is determined by a **deterministic ordering function**:

1. **Compare clocks** — lower clock means "earlier" insertion (logical time). Here, B's clock (7) < A's clock (10), so B goes first.
2. **If clocks are equal**, compare `clientID` — lower ID is placed earlier.

Because this rule is applied identically on every peer, **all peers independently compute the same ordering**. No central arbiter is needed.

#### 3. Concurrent Deletions

When one user deletes content while another edits it, Yjs marks the deleted content as a **tombstone** (soft delete). The tombstone remains in the data structure until all peers have acknowledged the deletion (tracked via vector clocks). This ensures:

- **No operation is ever lost** — even if two users completely overwrite each other, both edits exist in history
- The conflict resolution only determines **display order** — both edits survive

#### 4. Merging After Offline Edits

When a client reconnects after making offline changes:

```
Client                                 Server
  │                                      │
  │──── Sync Step 1 (state vector) ────→│  ← "Here's what I've seen"
  │←─── Sync Step 2 (missing ops) ─────│  ← "Here's what you missed"
  │←─── Your offline ops applied ──────│  ← CRDT merge
  │                                      │
  │         Both converge to same state  │
```

1. Client sends its **state vector** (summary of all operations it has seen).
2. Server responds with all operations the client is missing.
3. Server simultaneously applies the client's offline operations.
4. Because the merge function is **commutative and associative**, both sides reach the **exact same state**.

> 💡 **Visual analogy:** Imagine two people writing on the same whiteboard with different colored pens. Instead of writing on the board itself (which would collide), each person writes on their own transparent overlay. A rule determines which overlay's marks appear on top when they overlap. Yjs's rule: *lower `(clientID, clock)` goes first.*

### Why CRDTs over OT (Operational Transformation)?

| Aspect | CRDT (Yjs) | OT (Google Docs style) |
|--------|------------|----------------------|
| **Architecture** | Peer-to-peer or client-server | Requires central server |
| **Conflict model** | Automatic — no resolution needed | Requires transforms + ordering |
| **Offline** | Natural — merge on reconnect | Needs re-integration logic |
| **Undo** | Straightforward (per-peer undo stack) | Complex (must transform undo ops) |
| **Scalability** | Scales to many peers | Server becomes bottleneck |

---

## 🏗️ Architecture

```
┌─ Browser Tab 1 ─────┐     ┌─ Browser Tab 2 ─────┐
│ CodeMirror 6 Editor  │     │ CodeMirror 6 Editor  │
│ Yjs Y.Text (CRDT)   │     │ Yjs Y.Text (CRDT)   │
│ y-websocket client  │     │ y-websocket client  │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         └──────── WebSocket ────────┘
                      │
            ┌─────────▼──────────┐
            │  Node.js Server    │
            │  ws + yjs +      │
            │  y-protocols      │
            │  Room Manager     │
            └────────────────────┘
```

### Data Flow

```
Keystroke → CodeMirror → yCollab plugin → Y.Text update
    → Y.Doc 'update' event → WebsocketProvider → WebSocket
    → Server broadcasts to other peers → Y.Doc update
    → ySync plugin → CodeMirror DOM update
```

### Connection States

```
disconnected → connecting → connected → synced (full sync)
                                   ↓
                            disconnected (on close)
                                   ↓
                            exponential backoff reconnect
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** >= 18
- **npm** >= 9

### Installation

```bash
# 1. Install server dependencies
cd server
npm install

# 2. Install client dependencies
cd ../client
npm install
```

### Running

```bash
# Terminal 1 — Start the WebSocket server
cd server
npm start
# → Listening on http://localhost:1234

# Terminal 2 — Start the client dev server
cd client
npm run dev
# → Vite running at http://localhost:5173
```

### Try It

1. Open **http://localhost:5173** in your browser
2. Enter your name and click "Join Editor"
3. Open a **second browser tab** to the same URL
4. Enter a different name
5. Start typing in either tab — changes appear in both instantly!
6. Watch colored cursors showing where each person is editing

### Custom Rooms

Add `?room=my-room-name` to the URL to create isolated documents:

```
http://localhost:5173/?room=project-alpha
http://localhost:5173/?room=meeting-notes
```

---

## 🧩 Project Structure

```
collaborative-editor/
├── README.md                 # This file
├── package.json              # Root scripts
├── server/
│   ├── package.json
│   └── src/
│       └── index.js          # WebSocket + HTTP server (room mgmt, Yjs sync)
├── client/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx          # React entry point
│       ├── App.jsx           # Root component (orchestrator)
│       ├── hooks/
│       │   ├── useYjs.js     # Y.Doc + WebsocketProvider + UndoManager lifecycle
│       │   └── useAwareness.js # Awareness protocol (users, cursors)
│       ├── components/
│       │   ├── Editor.jsx           # CodeMirror 6 + yCollab
│       │   ├── Presence.jsx         # Online users + avatars
│       │   ├── ConnectionStatus.jsx # Connection indicator
│       │   └── UserNameDialog.jsx   # Name prompt overlay
│       ├── utils/
│       │   └── constants.js    # WS URL, room config, colors
│       └── styles/
│           ├── editor.css      # CodeMirror + app styles
│           ├── presence.css    # User avatars
│           └── connection.css  # Status indicators
```

### Key Packages

| Package | Purpose |
|---------|---------|
| **yjs** | CRDT library — the core data structure |
| **y-websocket** | WebSocket provider for Yjs (client) |
| **y-protocols** | Sync + Awareness protocol implementation (server) |
| **y-codemirror.next** | CodeMirror 6 binding for Yjs |
| **codemirror** | Text editor surface |
| **ws** | WebSocket server |
| **express** | HTTP server (health endpoint) |

---

## 🧪 Verification Tests

### 1. Single-user smoke test
- Open `http://localhost:5173`
- ✓ Editor loads with placeholder text
- ✓ Connection indicator shows "Connected" then "Synced"
- ✓ Can type and edit normally

### 2. Multi-user sync test
- Open two browser tabs to the same room
- ✓ Typing in tab 1 appears in tab 2 within milliseconds
- ✓ Both cursors visible (colored carets with user names)
- ✓ User count shows "2 online"

### 3. Conflict resolution test
- Place both cursors at the same word
- Tab 1 types "hello" and Tab 2 types "world" simultaneously
- ✓ Both appear — deterministic ordering based on CRDT rules

### 4. Offline / reconnect test
- Make edits in Tab 1
- Stop the server (`Ctrl+C`)
- ✓ Indicator shows "Disconnected (edits saved locally)"
- Continue typing in Tab 1
- Restart the server
- ✓ Indicator returns to "Connected" → "Synced"
- ✓ Offline edits appear in the document

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `1234` | WebSocket server port |
| `HOST` | `localhost` | WebSocket server host |
| `VITE_WS_URL` | `ws://localhost:1234` | WebSocket URL (client) |

---

## 📚 Further Reading

- [Yjs Documentation](https://docs.yjs.dev/)
- [Yjs — A CRDT for the Web (paper)](https://www.researchgate.net/publication/372576528_Yjs_A_CRDT_for_the_Web)
- [CRDTs Explained (Martin Kleppmann)](https://martin.kleppmann.com/papers/crdt.html)
- [y-codemirror.next](https://github.com/yjs/y-codemirror.next)
- [y-websocket](https://github.com/yjs/y-websocket)

---

## 📄 License

MIT
