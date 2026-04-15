// src/components/VehicleIdCard.jsx
import { motion, AnimatePresence } from 'framer-motion';

export default function VehicleIdCard({ plate, authorized, gateStatus, timestamp }) {
  const isOpen   = gateStatus === 'OPEN';
  const isDenied = gateStatus === 'DENIED';
  const active   = isOpen || isDenied;

  const accentColor = isOpen ? '#00ff88' : isDenied ? '#ff3860' : 'rgba(0,245,255,0.3)';
  const glow        = isOpen ? '0 0 30px rgba(0,255,136,0.3)' : isDenied ? '0 0 30px rgba(255,56,96,0.3)' : 'none';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <motion.div
        animate={active ? { boxShadow: glow, borderColor: accentColor } : { boxShadow: 'none', borderColor: 'rgba(0,245,255,0.15)' }}
        transition={{ duration: 0.4 }}
        style={{
          borderRadius: 10, border: '1px solid rgba(0,245,255,0.15)',
          padding: '1rem', background: 'rgba(0,245,255,0.03)',
          position: 'relative', display: 'flex', flexDirection: 'column', gap: '0.5rem',
        }}
      >
        {/* Stamp */}
        <AnimatePresence>
          {active && (
            <motion.div
              key={gateStatus}
              initial={{ opacity: 0, rotate: 6, scale: 0.7 }}
              animate={{ opacity: 1, rotate: 8, scale: 1 }}
              exit={{    opacity: 0, scale: 0.7 }}
              transition={{ type: 'spring', stiffness: 280, damping: 18 }}
              style={{
                position: 'absolute', top: 10, right: 12,
                fontFamily: "'Orbitron',sans-serif", fontWeight: 900,
                fontSize: '0.75rem', letterSpacing: '0.12em',
                border: `2px solid ${accentColor}`,
                color: accentColor, padding: '2px 8px', borderRadius: 4,
                textShadow: `0 0 8px ${accentColor}`,
              }}
            >
              {isOpen ? 'VERIFIED' : 'DENIED'}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Plate number */}
        <div style={{
          fontFamily: "'Orbitron',sans-serif", fontWeight: 900,
          fontSize: '1.4rem', letterSpacing: '0.15em',
          textAlign: 'center', color: '#fff',
          textShadow: `0 0 20px ${accentColor}`,
        }}>
          {plate || '—'}
        </div>

        {/* Status */}
        <div style={{
          textAlign: 'center', fontSize: '0.68rem', letterSpacing: '0.2em',
          color: active ? accentColor : 'rgba(0,245,255,0.4)',
          transition: 'color .3s',
        }}>
          {isOpen ? '✅ ACCESS GRANTED' : isDenied ? '❌ ACCESS DENIED' : 'AWAITING SCAN'}
        </div>

        {/* Meta rows */}
        <div style={{ borderTop: '1px solid rgba(0,245,255,0.1)', paddingTop: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
          {[
            ['TIME',   timestamp || '—'],
            ['METHOD', plate === 'Li-Fi Override' || plate === 'FORCE OVERRIDE' ? 'Li-Fi / Override' : 'OCR'],
          ].map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'rgba(0,245,255,0.45)' }}>
              <span>{k}</span>
              <span style={{ color: 'var(--cyan)' }}>{v}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* OCR readout */}
      <div style={{
        border: '1px solid rgba(0,245,255,0.12)', borderRadius: 8,
        padding: '0.5rem 0.8rem', background: 'rgba(0,245,255,0.02)',
      }}>
        <div style={{ fontSize: '0.58rem', letterSpacing: '0.18em', color: 'rgba(0,245,255,0.38)', marginBottom: 3 }}>OCR READING</div>
        <div style={{
          fontFamily: "'Orbitron',sans-serif", fontSize: '1.05rem', letterSpacing: '0.12em',
          color: authorized ? '#00ff88' : '#ff00c8',
          textShadow: authorized ? '0 0 10px #00ff88' : '0 0 10px #ff00c8',
          transition: 'color .2s',
        }}>
          {plate || '—'}
        </div>
      </div>
    </div>
  );
}
