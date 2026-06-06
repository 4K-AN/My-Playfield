import { useEffect, useRef, useState, useCallback } from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { WEBSOCKET_URL } from '../utils/constants.js';

/**
 * useYjs — manages a Yjs document, WebSocket provider, and undo manager.
 *
 * Returns:
 *   ydoc           — the Y.Doc instance (read-only after mount)
 *   provider       — the WebsocketProvider instance
 *   ytext          — the shared Y.Text type named 'codemirror'
 *   undoManager    — the Y.UndoManager bound to ytext
 *   awareness      — provider.awareness (Awareness protocol)
 *   connectionStatus — 'connecting' | 'connected' | 'disconnected'
 *   isSynced       — boolean, true after initial sync completes
 *   connect        — () => void, manually connect
 *   disconnect     — () => void, manually disconnect
 */
export function useYjs(roomName) {
  // Stable refs — created once, survive re-renders
  const ydocRef = useRef(null);
  const providerRef = useRef(null);
  const ytextRef = useRef(null);
  const undoManagerRef = useRef(null);

  // Reactive state
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [isSynced, setIsSynced] = useState(false);
  const [awareness, setAwareness] = useState(null);

  // ── Initialise on mount ──
  useEffect(() => {
    const ydoc = new Y.Doc();
    ydocRef.current = ydoc;

    const ytext = ydoc.getText('codemirror');
    ytextRef.current = ytext;

    const url = WEBSOCKET_URL;
    const provider = new WebsocketProvider(url, roomName, ydoc, {
      connect: true,
      maxBackoffTime: 5000,
    });
    providerRef.current = provider;
    setAwareness(provider.awareness);

    // Connection status events
    const unsubStatus = provider.on('status', ({ status }) => {
      setConnectionStatus(status);
    });

    // Sync state — true after step 2 received (full sync)
    const unsubSync = provider.on('sync', (synced) => {
      setIsSynced(synced);
    });

    // Undo manager (track only local edits by default)
    const um = new Y.UndoManager(ytext, {
      trackedOrigins: new Set([null]),
      captureTimeout: 500,
    });
    undoManagerRef.current = um;

    // ── Cleanup on unmount ──
    return () => {
      unsubStatus();
      unsubSync();
      um.destroy();
      provider.destroy();
      ydoc.destroy();

      ydocRef.current = null;
      providerRef.current = null;
      ytextRef.current = null;
      undoManagerRef.current = null;
    };
  }, [roomName]);

  // ── Manual connect / disconnect ──
  const connect = useCallback(() => {
    providerRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    providerRef.current?.disconnect();
  }, []);

  return {
    ydoc: ydocRef.current,
    provider: providerRef.current,
    ytext: ytextRef.current,
    undoManager: undoManagerRef.current,
    awareness,
    connectionStatus,
    isSynced,
    connect,
    disconnect,
  };
}
