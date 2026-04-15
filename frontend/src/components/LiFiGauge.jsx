// src/components/LiFiGauge.jsx
// Derived from 21st-dev Gauge component (SVG arc, custom color thresholds)
import { useMemo } from 'react';

const STROKE_W  = 10;
const RADIUS    = 45;
const CX        = 50;
const CIRC      = 2 * Math.PI * RADIUS;

function getColor(pct) {
  if (pct >= 0.79) return '#00ff88';
  if (pct >= 0.40) return '#00f5ff';
  return '#ff3860';
}

export default function LiFiGauge({ intensity = 0, state = false, pulseCount = 0 }) {
  const pct   = Math.min(intensity / 255, 1);
  const color = getColor(pct);
  const dash  = useMemo(() => CIRC * pct, [pct]);
  const gap   = CIRC - dash;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem' }}>
      {/* SVG circular gauge */}
      <svg viewBox="0 0 100 100" width={160} height={160} style={{ overflow: 'visible' }}>
        {/* Track */}
        <circle cx={CX} cy={CX} r={RADIUS}
          fill="none" stroke="rgba(0,245,255,0.1)" strokeWidth={STROKE_W} />
        {/* Fill — rotated so it starts at top */}
        <circle cx={CX} cy={CX} r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE_W}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap}`}
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: '50px 50px',
            filter:   `drop-shadow(0 0 8px ${color})`,
            transition: 'stroke-dasharray 0.35s ease, stroke 0.35s ease',
          }}
        />
        {/* Center text */}
        <text x="50" y="46" textAnchor="middle" dominantBaseline="middle"
          fill={color} fontSize={20} fontFamily="'Orbitron',sans-serif" fontWeight="bold">
          {Math.round(intensity)}
        </text>
        <text x="50" y="62" textAnchor="middle" dominantBaseline="middle"
          fill="rgba(0,245,255,0.4)" fontSize={7} fontFamily="'JetBrains Mono',monospace"
          letterSpacing="0.12em">
          INTENSITY
        </text>
      </svg>

      {/* State label */}
      <div style={{
        fontSize: '0.8rem', letterSpacing: '0.12em',
        color:      state ? '#00ff88' : '#ff3860',
        textShadow: state ? '0 0 12px #00ff88' : '0 0 8px #ff3860',
        transition: 'color .2s, text-shadow .2s',
      }}>
        ⬤ {state ? 'HIGH' : 'LOW'}
      </div>

      {/* Pulse dots */}
      <div style={{ display: 'flex', gap: '0.9rem' }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 26, height: 26, borderRadius: '50%',
            border: `2px solid ${i < pulseCount ? '#00ff88' : 'rgba(0,245,255,0.2)'}`,
            background: i < pulseCount ? '#00ff88' : 'transparent',
            boxShadow:  i < pulseCount ? '0 0 14px #00ff88' : 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '0.65rem', color: i < pulseCount ? '#000' : 'rgba(0,245,255,0.4)',
            fontWeight: '600',
            transition: 'all .2s',
          }}>
            {i + 1}
          </div>
        ))}
      </div>
      <div style={{ fontSize: '0.58rem', letterSpacing: '0.18em', color: 'rgba(0,245,255,0.35)' }}>
        PULSES (3 = UNLOCK)
      </div>
    </div>
  );
}
