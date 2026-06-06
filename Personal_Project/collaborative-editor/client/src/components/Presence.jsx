/**
 * Presence — shows online user avatars and count.
 */
export default function Presence({ users }) {
  // Don't show anything if only the local user is present
  const others = users.filter((u) => !u.isLocal);
  const totalCount = users.length;

  if (totalCount === 0) return null;

  return (
    <div className="presence-bar">
      <span className="online-count">
        <span className="dot-green" />
        {totalCount} {totalCount === 1 ? 'online' : 'online'}
      </span>

      <div className="user-avatars">
        {users.map((user) => (
          <div
            key={user.clientId}
            className="user-avatar"
            style={{ backgroundColor: user.color }}
            title={`${user.name}${user.isLocal ? ' (you)' : ''}`}
          >
            <span className="user-initial">
              {user.name.charAt(0).toUpperCase()}
            </span>
            {user.isLocal && <span className="self-badge">you</span>}
          </div>
        ))}

        {others.length > 0 && (
          <span className="others-label">
            {others.map((u) => u.name).join(', ')}
          </span>
        )}
      </div>
    </div>
  );
}
