// src/components/AlertBanner.jsx
// Adapted from 21st-dev Alert + Banner pattern with a cyberpunk glitch effect
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

export default function AlertBanner({ gateStatus, plate, timestamp }) {
  const [visible, setVisible] = useState(false);
  const isOpen = gateStatus === 'OPEN';
  const isDenied = gateStatus === 'DENIED';

  useEffect(() => {
    if (isOpen || isDenied) {
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 6000);
      return () => clearTimeout(t);
    }
  }, [gateStatus, plate]);

  const color  = isOpen ? '#00ff88' : '#ff3860';
  const border = isOpen ? 'rgba(0,255,136,0.35)' : 'rgba(255,56,96,0.35)';
  const glow   = isOpen ? '0 0 30px rgba(0,255,136,0.45)' : '0 0 30px rgba(255,56,96,0.45)';

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key={plate + gateStatus}
          initial={{ opacity: 0, y: -30, scale: 0.95 }}
          animate={{ opacity: 1, y: 0,   scale: 1 }}
          exit={{    opacity: 0, y: -20,  scale: 0.95 }}
          transition={{ duration: 0.3, type: 'spring', stiffness: 260, damping: 20 }}
          style={{
            position: 'relative', overflow: 'hidden',
            background: `linear-gradient(135deg, rgba(5,5,16,0.9), rgba(5,5,16,0.95))`,
            border: `1px solid ${border}`,
            borderRadius: 10, padding: '0.7rem 1.2rem',
            boxShadow: glow,
          }}
        >
          {/* Glitch overlay — renders on top, loops twice then stops */}
          <div style={{
            position: 'absolute', inset: 0, color,
            fontFamily: "'Orbitron',sans-serif",
            fontWeight: 900, fontSize: '1.1rem',
            letterSpacing: '0.15em',
            display: 'flex', alignItems: 'center', pointerEvents: 'none',
            animation: 'glitch 0.35s steps(1) 2',
            opacity: 0.25,
          }}>
            {isOpen ? '⚡ ACCESS GRANTED' : '✖ ACCESS DENIED'}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '1.5rem' }}>{isOpen ? '✅' : '❌'}</span>
            <div>
              <div style={{
                fontFamily: "'Orbitron',sans-serif", fontWeight: 900,
                fontSize: '0.9rem', color, letterSpacing: '0.14em',
                textShadow: `0 0 12px ${color}`,
              }}>
                {isOpen ? '⚡ ACCESS GRANTED' : '✖ ACCESS DENIED'}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'rgba(0,245,255,0.6)', marginTop: 2 }}>
                {plate || '—'} &nbsp;·&nbsp; {timestamp || ''}
              </div>
            </div>
            <button
              onClick={() => setVisible(false)}
              style={{
                marginLeft: 'auto', background: 'none', border: 'none',
                color: 'rgba(0,245,255,0.4)', cursor: 'pointer', fontSize: '1rem',
              }}
            >✕</button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
