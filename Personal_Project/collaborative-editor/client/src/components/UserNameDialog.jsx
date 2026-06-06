import { useState } from 'react';
import { getColorForName } from '../utils/constants.js';

/**
 * UserNameDialog — prompts for a username on first visit.
 *
 * Props:
 *   onSetName  — (name: string) => void, called when user submits
 */
export default function UserNameDialog({ onSetName }) {
  const [name, setName] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed.length < 1) return;
    setSubmitted(true);
    onSetName(trimmed);
  };

  // Allow quick reuse — if already submitted, don't show
  if (submitted) return null;

  const previewColor = name.trim()
    ? getColorForName(name.trim())
    : '#ccc';

  return (
    <div className="user-name-overlay">
      <div className="user-name-dialog">
        <h2>Welcome to the Collaborative Editor</h2>
        <p className="subtitle">
          Choose a display name so others can see you.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <div
              className="avatar-preview"
              style={{ backgroundColor: previewColor }}
            >
              {name.trim() ? name.trim().charAt(0).toUpperCase() : '?'}
            </div>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name…"
              maxLength={24}
              autoFocus
              className="name-input"
            />
          </div>
          <button
            type="submit"
            className="join-btn"
            disabled={!name.trim()}
          >
            Join Editor
          </button>
        </form>
      </div>
    </div>
  );
}
