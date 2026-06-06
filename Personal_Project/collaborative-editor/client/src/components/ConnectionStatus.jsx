/**
 * ConnectionStatus — visual indicator for WebSocket connection state.
 *
 * Props:
 *   status    — 'connecting' | 'connected' | 'disconnected'
 *   isSynced  — boolean
 */
export default function ConnectionStatus({ status, isSynced }) {
  const config = {
    connected: {
      className: 'status-connected',
      label: 'Connected',
      dot: 'dot-green',
    },
    connecting: {
      className: 'status-connecting',
      label: 'Connecting…',
      dot: 'dot-yellow',
    },
    disconnected: {
      className: 'status-disconnected',
      label: 'Disconnected',
      dot: 'dot-red',
    },
  };

  const { className, label, dot } = config[status] || config.disconnected;

  return (
    <div className={`connection-status ${className}`}>
      <span className={`status-dot ${dot}`} />
      <span className="status-label">{label}</span>
      {status === 'connected' && !isSynced && (
        <span className="syncing-indicator">Syncing…</span>
      )}
      {status === 'disconnected' && (
        <span className="offline-hint">(edits saved locally)</span>
      )}
    </div>
  );
}
