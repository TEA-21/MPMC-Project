// src/components/TacticalVideo.jsx
export default function TacticalVideo({ plate }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', flex: 1 }}>
      {/* Stream wrapper */}
      <div style={{
        position: 'relative', borderRadius: 8, overflow: 'hidden',
        border: '1px solid rgba(0,245,255,0.2)',
        background: '#000', flex: 1, minHeight: 280,
      }}>
        <img
          src="/video_feed"
          alt="Live feed"
          style={{ width: '100%', display: 'block', objectFit: 'cover', height: '100%' }}
        />
        {/* Scanning line */}
        <div style={{
          position: 'absolute', left: 0, right: 0, height: 3,
          background: 'linear-gradient(90deg, transparent, #00f5ff, transparent)',
          opacity: 0.65, animation: 'scan 2.8s linear infinite', pointerEvents: 'none',
        }} />
        {/* Corner: REC */}
        <div style={{
          position: 'absolute', top: 8, left: 10,
          color: '#ff3860', fontSize: '0.65rem', letterSpacing: '0.1em',
          background: 'rgba(0,0,0,0.55)', padding: '2px 8px', borderRadius: 4,
          animation: 'blink 1.2s infinite',
        }}>● REC</div>
        {/* Corner: plate */}
        <div style={{
          position: 'absolute', top: 8, right: 10,
          color: '#00f5ff', fontSize: '0.8rem', fontWeight: 600,
          background: 'rgba(0,0,0,0.55)', padding: '2px 10px', borderRadius: 4,
          textShadow: '0 0 8px #00f5ff', letterSpacing: '0.08em',
        }}>{plate || '—'}</div>
      </div>
    </div>
  );
}
