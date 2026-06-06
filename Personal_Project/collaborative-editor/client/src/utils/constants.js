// ─── WebSocket server connection ───────────────────────────────────
export const WEBSOCKET_URL =
  import.meta.env.VITE_WS_URL || 'ws://localhost:1234';

// ─── Room configuration ──────────────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
export const ROOM_NAME = urlParams.get('room') || 'default';

// ─── User color palette (deterministic from name hash) ────────────
export const USER_COLORS = [
  '#30bced',
  '#ff6b6b',
  '#ffa94d',
  '#69db7c',
  '#9775fa',
  '#f783ac',
  '#74c0fc',
  '#ffd43b',
  '#20c997',
  '#845ef7',
];

export function getColorForName(name, light = false) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  const idx = Math.abs(hash) % USER_COLORS.length;
  const base = USER_COLORS[idx];
  if (light) return base + '44'; // 27% opacity for selection bg
  return base;
}
