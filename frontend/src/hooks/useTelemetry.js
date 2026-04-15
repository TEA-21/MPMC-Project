// src/hooks/useTelemetry.js
import { useEffect, useState, useCallback } from 'react';
import { io } from 'socket.io-client';

const SOCKET_URL = '/'; // Vite proxy handles routing to :8000

let _socket = null;
function getSocket() {
  if (!_socket) {
    // Start with polling (always works behind a Vite proxy), then upgrade to WS.
    // Reversing this order ('websocket' first) can cause silent failures when the
    // WS upgrade is delayed by the proxy while the HTTP port is already open.
    _socket = io(SOCKET_URL, {
      transports: ['polling', 'websocket'],
      reconnectionDelay: 1000,
      reconnectionAttempts: Infinity,
    });
  }
  return _socket;
}

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState({
    intensity:   0,
    pulse_count: 0,
    lifi_state:  false,
    plate_text:  '',
    gate_status: 'IDLE',
    authorized:  false,
    last_gate_ts: '',
  });
  const [logs, setLogs]       = useState([]);
  const [connected, setConnected] = useState(false);

  const addLog = useCallback((message, level = 'info') => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs(prev => [...prev.slice(-150), { id: Date.now() + Math.random(), ts, message, level }]);
  }, []);

  useEffect(() => {
    const socket = getSocket();

    socket.on('connect',    () => { setConnected(true);  addLog('[SOCKET] Connected to Edge-AI LPR.', 'info'); });
    socket.on('disconnect', () => { setConnected(false); addLog('[SOCKET] Disconnected.', 'error'); });

    socket.on('state_snapshot', data => setTelemetry(t => ({ ...t, ...data })));

    socket.on('telemetry_update', data => setTelemetry(t => ({ ...t, ...data })));

    socket.on('gate_event', data => {
      setTelemetry(t => ({
        ...t,
        gate_status:  data.status,
        plate_text:   data.plate,
        last_gate_ts: data.timestamp,
        authorized:   data.status === 'OPEN',
      }));
      addLog(`[GATE] ${data.status} — ${data.plate} @ ${data.timestamp}`,
             data.status === 'OPEN' ? 'success' : 'error');
    });

    socket.on('log_line', data => addLog(data.message, data.level || 'info'));

    socket.on('system_reset', () => {
      setTelemetry({ intensity: 0, pulse_count: 0, lifi_state: false,
                     plate_text: '', gate_status: 'IDLE', authorized: false, last_gate_ts: '' });
      addLog('[SYSTEM] Reset complete.', 'info');
    });

    return () => {
      socket.off('connect'); socket.off('disconnect');
      socket.off('state_snapshot'); socket.off('telemetry_update');
      socket.off('gate_event'); socket.off('log_line'); socket.off('system_reset');
    };
  }, [addLog]);

  return { telemetry, logs, connected };
}
