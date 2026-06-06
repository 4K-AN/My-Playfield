import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, placeholder } from '@codemirror/view';
import { javascript } from '@codemirror/lang-javascript';
import { yCollab } from 'y-codemirror.next';

/**
 * Editor — CodeMirror 6 bound to a Yjs shared text type.
 *
 * Props:
 *   ytext         — Y.Text (from Y.Doc)
 *   awareness     — from WebsocketProvider.awareness
 *   undoManager   — Y.UndoManager instance
 *   userName      — string, displayed in remote-cursor tooltips
 *   connectionStatus — for read-only state when disconnected
 */
export default function Editor({
  ytext,
  awareness,
  undoManager,
  userName,
  connectionStatus,
}) {
  const editorRef = useRef(null);
  const viewRef = useRef(null);

  useEffect(() => {
    if (!ytext || !editorRef.current) return;

    // yCollab combines:
    //   - ySync (bidirectional Y.Text ↔ CodeMirror binding)
    //   - yRemoteSelections (remote cursors + selections)
    //   - yUndoManager (Yjs-aware undo/redo)
    const collabExt = yCollab(ytext, awareness, { undoManager });

    const extensions = [
      // CodeMirror essentials
      EditorView.theme({
        '&': { height: '100%' },
        '.cm-scroller': { overflow: 'auto' },
        '.cm-content': { fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace", fontSize: '14px', lineHeight: '1.6' },
        '.cm-gutters': { fontFamily: "'JetBrains Mono', Consolas, monospace", fontSize: '12px' },
      }),
      placeholder('Start typing together…'),
      javascript(),

      // Yjs CRDT collab
      collabExt,

      // Keyboard shortcuts
      keymap.of([
        { key: 'Mod-z', run: () => { undoManager?.undo(); return true; } },
        { key: 'Mod-Shift-z', run: () => { undoManager?.redo(); return true; } },
        { key: 'Mod-y', run: () => { undoManager?.redo(); return true; } },
      ]),

      // Update listener — broadcast cursor position via awareness
      EditorView.updateListener.of((update) => {
        if (update.selectionSet) {
          const sel = update.state.selection.main;
          if (awareness && userName) {
            awareness.setLocalStateField('cursor', {
              anchor: { ytext: ytext, index: sel.anchor },
              head: { ytext: ytext, index: sel.head },
            });
          }
        }
      }),
    ];

    const state = EditorState.create({
      doc: ytext.toString(),
      extensions,
    });

    const view = new EditorView({
      state,
      parent: editorRef.current,
    });
    viewRef.current = view;

    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, [ytext, awareness, undoManager, userName]);

  return (
    <div
      className={`editor-wrapper ${connectionStatus === 'disconnected' ? 'editor-offline' : ''}`}
      ref={editorRef}
    />
  );
}
