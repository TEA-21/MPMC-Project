// src/App.jsx — Cyberpunk Security Command Center
import { useState, useEffect } from 'react';
import { useTelemetry }   from './hooks/useTelemetry';
import LiFiGauge          from './components/LiFiGauge';
import TacticalVideo      from './components/TacticalVideo';
import AlertBanner        from './components/AlertBanner';
import HackerTerminal     from './components/HackerTerminal';
import VehicleIdCard      from './components/VehicleIdCard';

// ── Panel wrapper ──────────────────────────────────────────────────────────
function Panel({ title, children, style }) {
  return (
    <div style={{
      background: 'rgba(0,245,255,0.04)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(0,245,255,0.15)',
      borderRadius: 12, padding: '1rem',
      display: 'flex', flexDirection: 'column', gap: '0.75rem',
      ...style,
    }}>
      <h2 style={{
        fontFamily: "'Orbitron',sans-serif", fontSize: '0.62rem',
        letterSpacing: '0.22em', color: '#00f5ff',
        textShadow: '0 0 8px #00f5ff',
        paddingBottom: '0.5rem',
        borderBottom: '1px solid rgba(0,245,255,0.14)',
      }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

// ── Gate Status Bar ────────────────────────────────────────────────────────
function GateBar({ status, plate, ts }) {
  const color = status === 'OPEN' ? '#00ff88' : status === 'DENIED' ? '#ff3860' : '#00f5ff';
  const glow  = status === 'OPEN'
    ? { animation: 'gatePulse 1.5s ease-in-out infinite', border: '1px solid #00ff88' }
    : status === 'DENIED'
    ? { border: '1px solid #ff3860', boxShadow: '0 0 18px rgba(255,56,96,0.4)' }
    : { border: '1px solid rgba(0,245,255,0.15)' };

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.6rem',
      padding: '0.5rem 1rem', borderRadius: 8,
      background: 'rgba(0,245,255,0.03)',
      transition: 'border .3s, box-shadow .3s',
      ...glow,
    }}>
      <span style={{
        width: 10, height: 10, borderRadius: '50%',
        background: color, boxShadow: `0 0 8px ${color}`,
        display: 'inline-block', flexShrink: 0,
      }} />
      <span style={{ fontFamily: "'Orbitron',sans-serif", fontSize: '0.72rem', color, flex: 1, letterSpacing: '0.1em' }}>
        GATE: {status}
      </span>
      <span style={{ fontSize: '0.62rem', color: 'rgba(0,245,255,0.45)' }}>{ts}</span>
    </div>
  );
}

// ── Neon Button ────────────────────────────────────────────────────────────
function NeonBtn({ label, onClick, variant = 'open' }) {
  const isCyan = variant === 'open';
  const c = isCyan ? '#00f5ff' : '#ff00c8';
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1, padding: '0.65rem 0.5rem',
        fontFamily: "'Orbitron',sans-serif", fontSize: '0.62rem',
        letterSpacing: '0.1em', textTransform: 'uppercase',
        border: `1px solid ${c}`, borderRadius: 8, cursor: 'pointer',
        background: isCyan ? 'linear-gradient(135deg,#002233,#004455)' : 'linear-gradient(135deg,#220022,#440044)',
        color: c, boxShadow: `0 4px 0 ${isCyan?'#001122':'#110011'}, 0 0 16px ${c}33`,
        transition: 'box-shadow .2s',
      }}
      onMouseEnter={e => e.currentTarget.style.boxShadow = `0 4px 0 ${isCyan?'#001122':'#110011'}, 0 0 32px ${c}88`}
      onMouseLeave={e => e.currentTarget.style.boxShadow = `0 4px 0 ${isCyan?'#001122':'#110011'}, 0 0 16px ${c}33`}
    >
      {label}
    </button>
  );
}

// ── Live clock ─────────────────────────────────────────────────────────────
function useClock() {
  const [t, setT] = useState('');
  useEffect(() => {
    const tick = () => setT(new Date().toTimeString().slice(0, 8));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return t;
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const { telemetry, logs, connected } = useTelemetry();
  const clock = useClock();

  const handleForceOpen  = () => fetch('/force-open',  { method: 'POST' });
  const handleSystemReset = () => fetch('/system-reset', { method: 'POST' });

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>

      {/* ── HEADER ── */}
      <header style={{
        display: 'flex', alignItems: 'center', gap: '1.5rem',
        padding: '0.65rem 2rem',
        background: 'linear-gradient(90deg, rgba(0,245,255,0.08), transparent 60%)',
        borderBottom: '1px solid rgba(0,245,255,0.15)',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: '1.5rem' }}>⚡</span>
        <span style={{ fontFamily: "'Orbitron',sans-serif", fontWeight: 900, fontSize: '1.2rem', letterSpacing: '0.15em', textShadow: '0 0 16px #00f5ff' }}>
          EDGE<span style={{ color: '#ff00c8', textShadow: '0 0 16px #ff00c8' }}>-AI</span> LPR
        </span>
        <span style={{ flex: 1, fontSize: '0.65rem', letterSpacing: '0.18em', color: 'rgba(0,245,255,0.4)' }}>
          CYBERPUNK SECURITY COMMAND CENTER
        </span>
        <span style={{
          fontSize: '0.65rem', padding: '2px 10px', borderRadius: 12,
          border: `1px solid ${connected ? '#00ff88' : '#ff3860'}`,
          color: connected ? '#00ff88' : '#ff3860',
        }}>
          ● {connected ? 'ONLINE' : 'OFFLINE'}
        </span>
        <span style={{ fontFamily: "'Orbitron',sans-serif", fontSize: '0.9rem', color: '#00f5ff', textShadow: '0 0 10px #00f5ff' }}>
          {clock}
        </span>
      </header>

      {/* ── ALERT BANNER ── */}
      <div style={{ padding: '0 1.5rem', paddingTop: '0.5rem' }}>
        <AlertBanner
          gateStatus={telemetry.gate_status}
          plate={telemetry.plate_text}
          timestamp={telemetry.last_gate_ts}
        />
      </div>

      {/* ── MAIN 3-COL GRID ── */}
      <main style={{
        display: 'grid', gridTemplateColumns: '220px 1fr 250px',
        gap: '1rem', padding: '0.75rem 1.5rem', flex: 1, minHeight: 0,
      }}>

        {/* Col 1: Li-Fi */}
        <Panel title="LI-FI PULSE METER" style={{ alignItems: 'center' }}>
          <LiFiGauge
            intensity={telemetry.intensity}
            state={telemetry.lifi_state}
            pulseCount={telemetry.pulse_count}
          />
        </Panel>

        {/* Col 2: Video + Gate bar + Buttons */}
        <Panel title="LIVE TACTICAL STREAM" style={{ flex: 1 }}>
          <TacticalVideo plate={telemetry.plate_text} />
          <GateBar
            status={telemetry.gate_status}
            plate={telemetry.plate_text}
            ts={telemetry.last_gate_ts}
          />
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <NeonBtn label="⚡ FORCE OPEN"  onClick={handleForceOpen}   variant="open"  />
            <NeonBtn label="↺ SYSTEM RESET" onClick={handleSystemReset} variant="reset" />
          </div>
        </Panel>

        {/* Col 3: ID Card */}
        <Panel title="VEHICLE IDENTITY">
          <VehicleIdCard
            plate={telemetry.plate_text}
            authorized={telemetry.authorized}
            gateStatus={telemetry.gate_status}
            timestamp={telemetry.last_gate_ts}
          />
        </Panel>
      </main>

      {/* ── TERMINAL ── */}
      <div style={{ padding: '0 1.5rem 1rem' }}>
        <HackerTerminal logs={logs} />
      </div>
    </div>
  );
}
