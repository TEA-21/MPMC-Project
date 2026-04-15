// src/components/HackerTerminal.jsx
// Adapted from 21st-dev KineticLogStream — real LPR logs, no random data
import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const LEVEL_COLORS = {
  success: '#00ffcc',
  lifi:    '#00ff88',
  error:   '#ff3860',
  info:    '#00f5ff',
};

export default function HackerTerminal({ logs }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);

  return (
    <div style={{
      background: '#020208',
      borderRadius: 10,
      border: '1px solid rgba(0,245,255,0.12)',
      overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Title bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.4rem',
        padding: '0.4rem 0.9rem',
        background: 'rgba(0,245,255,0.06)',
        borderBottom: '1px solid rgba(0,245,255,0.1)',
      }}>
        {['#ff5f57','#febc2e','#28c840'].map((c, i) => (
          <span key={i} style={{ width: 10, height: 10, borderRadius: '50%', background: c, display: 'inline-block' }} />
        ))}
        <span style={{ marginLeft: '0.5rem', fontSize: '0.62rem', letterSpacing: '0.18em', color: 'rgba(0,245,255,0.4)' }}>
          SYSTEM LOG — EDGE-AI LPR v2.0
        </span>
      </div>

      {/* Scrollable log area */}
      <div ref={ref} style={{ height: 160, overflowY: 'auto', padding: '0.4rem 0.9rem' }}>
        <AnimatePresence initial={false}>
          {logs.map(log => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1,  x: 0 }}
              exit={{    opacity: 0 }}
              transition={{ duration: 0.18 }}
              style={{
                display: 'flex', gap: '0.75rem',
                fontSize: '0.7rem', lineHeight: 1.6,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              <span style={{ color: 'rgba(0,245,255,0.3)', flexShrink: 0 }}>[{log.ts}]</span>
              <span style={{ color: LEVEL_COLORS[log.level] || '#00f5ff' }}>{log.message}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
