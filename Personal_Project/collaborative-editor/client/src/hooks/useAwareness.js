import { useEffect, useState, useCallback } from 'react';
import { getColorForName } from '../utils/constants.js';

/**
 * useAwareness — subscribes to Yjs Awareness protocol and manages
 * local user state (name + cursor).
 *
 * Returns:
 *   users         — array of { clientId, name, color, colorLight, isLocal }
 *   updateCursor  — (anchor, head) => void, broadcast cursor position
 */
export function useAwareness(awareness, userName) {
  const [users, setUsers] = useState([]);

  // ── Subscribe to awareness changes ──
  useEffect(() => {
    if (!awareness) return;

    // Set local user state whenever userName changes
    if (userName) {
      awareness.setLocalStateField('user', {
        name: userName,
        color: getColorForName(userName),
        colorLight: getColorForName(userName, true),
      });
    }

    const updateUsers = () => {
      const states = awareness.getStates();
      const userList = [];
      const myId = awareness.doc.clientID;

      states.forEach((state, clientId) => {
        if (state.user && state.user.name) {
          userList.push({
            clientId,
            name: state.user.name,
            color: state.user.color || '#888',
            colorLight: state.user.colorLight || state.user.color + '44',
            isLocal: clientId === myId,
          });
        }
      });

      // Sort: local user first, then alphabetical
      userList.sort((a, b) => {
        if (a.isLocal) return -1;
        if (b.isLocal) return 1;
        return a.name.localeCompare(b.name);
      });

      setUsers(userList);
    };

    awareness.on('change', updateUsers);
    updateUsers();

    return () => {
      awareness.off('change', updateUsers);
      // Clear local awareness on unmount
      awareness.setLocalStateField('user', null);
      awareness.setLocalStateField('cursor', null);
    };
  }, [awareness, userName]);

  // ── Broadcast cursor position ──
  const updateCursor = useCallback(
    (anchor, head) => {
      if (!awareness) return;
      awareness.setLocalStateField('cursor', { anchor, head });
    },
    [awareness]
  );

  return { users, updateCursor };
}
