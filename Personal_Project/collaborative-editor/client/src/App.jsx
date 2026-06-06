import { useState, useEffect } from 'react';
import { useYjs } from './hooks/useYjs.js';
import { useAwareness } from './hooks/useAwareness.js';
import { ROOM_NAME } from './utils/constants.js';
import Editor from './components/Editor.jsx';
import Presence from './components/Presence.jsx';
import ConnectionStatus from './components/ConnectionStatus.jsx';
import UserNameDialog from './components/UserNameDialog.jsx';

const STORAGE_KEY = 'collab-editor-username';

export default function App() {
  // ── Persistent username ──
  const [userName, setUserName] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || '';
  });

  const handleSetName = (name) => {
    localStorage.setItem(STORAGE_KEY, name);
    setUserName(name);
  };

  const handleClearName = () => {
    localStorage.removeItem(STORAGE_KEY);
    setUserName('');
  };

  // ── Yjs document + provider ──
  const {
    ytext,
    undoManager,
    awareness,
    connectionStatus,
    isSynced,
  } = useYjs(ROOM_NAME);

  // ── Awareness (presence) ──
  const { users } = useAwareness(awareness, userName);

  // ── Room name from URL ──
  const [room, setRoom] = useState(ROOM_NAME);

  // ── Render ──
  const needsName = !userName;

  return (
    <div className="app">
      {/* Connection status bar */}
      <ConnectionStatus status={connectionStatus} isSynced={isSynced} />

      {/* Username prompt overlay */}
      {needsName && <UserNameDialog onSetName={handleSetName} />}

      {/* Main editor UI (hidden behind name dialog) */}
      <div className={`editor-container ${needsName ? 'editor-container--blurred' : ''}`}>
        <header className="toolbar">
          <h1 className="toolbar-title">
            <span className="toolbar-icon">📝</span>
            <span>Collaborative Editor</span>
          </h1>

          <Presence users={users} />

          <div className="toolbar-actions">
            <label className="room-label">
              Room:
              <input
                type="text"
                className="room-input"
                value={room}
                onChange={(e) => setRoom(e.target.value)}
                onBlur={() => {
                  const newRoom = room.trim() || 'default';
                  setRoom(newRoom);
                  window.history.replaceState(null, '', `?room=${encodeURIComponent(newRoom)}`);
                  window.location.reload();
                }}
              />
            </label>

            <button
              className="tool-btn"
              onClick={() => undoManager?.undo()}
              title="Undo (Ctrl+Z)"
            >
              ↩
            </button>
            <button
              className="tool-btn"
              onClick={() => undoManager?.redo()}
              title="Redo (Ctrl+Shift+Z)"
            >
              ↪
            </button>

            <div className="user-badge" onClick={handleClearName} title="Click to change name">
              <span
                className="user-badge-avatar"
                style={{ backgroundColor: userName ? undefined : '#888' }}
              >
                {userName ? userName.charAt(0).toUpperCase() : '?'}
              </span>
              <span className="user-badge-name">{userName || 'Unnamed'}</span>
            </div>
          </div>
        </header>

        <main className="editor-main">
          <Editor
            ytext={ytext}
            awareness={awareness}
            undoManager={undoManager}
            userName={userName}
            connectionStatus={connectionStatus}
          />
        </main>
      </div>
    </div>
  );
}
