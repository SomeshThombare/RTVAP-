import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDownLeft, ArrowUpRight, CarFront, CircleHelp, Radio, RefreshCw, Wifi } from 'lucide-react';
import './style.css';

// Use the Vite same-origin proxy so localhost and network-address browser tabs work.
const API = import.meta.env.VITE_API_URL || '';
const COLORS = ['white', 'black', 'silver', 'gray', 'blue', 'red', 'orange', 'yellow', 'green', 'purple', 'unknown'];
const colorHex = { white: '#d9ddd8', black: '#555d58', silver: '#aab4b4', gray: '#808984', blue: '#72a9f5', red: '#f27866', orange: '#f39b55', yellow: '#e6cf58', green: '#72bb8a', purple: '#ad8ae0', unknown: '#858b91' };
const fmt = (v) => new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(v));

function App() {
  const [metrics, setMetrics] = useState({ total: 0, by_color: [], by_direction: [], mode: 'demo' });
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const refresh = async () => {
    try {
      const [m, e] = await Promise.all([fetch(`${API}/api/metrics`), fetch(`${API}/api/events?limit=30`)]);
      if (!m.ok || !e.ok) throw new Error('API unavailable');
      setMetrics(await m.json()); setEvents(await e.json()); setError('');
    } catch { setError('API unavailable. Start the backend and PostgreSQL to connect.'); }
  };
  useEffect(() => {
    refresh();
    const source = new EventSource(`${API}/api/events/stream`);
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      setEvents((old) => [event, ...old.filter((row) => row.id !== event.id)].slice(0, 30));
      refresh();
    };
    // Poll as a fallback so the dashboard still reflects database writes if SSE is unavailable.
    const poll = window.setInterval(refresh, 5000);
    return () => { source.close(); window.clearInterval(poll); };
  }, []);
  const palette = useMemo(() => Object.fromEntries(metrics.by_color.map((x) => [x.color, x.count])), [metrics]);
  const directions = Object.fromEntries(metrics.by_direction.map((x) => [x.direction, x.count]));
  const maxColor = Math.max(1, ...COLORS.map((c) => palette[c] || 0));

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-mark"><Activity size={18}/></span><span>ROADWATCH<small>VISION SYSTEMS</small></span></a>
      <div className="top-meta"><span className={`live-pill ${connected || !error ? 'connected' : ''}`}><i/> {connected ? 'LIVE UPDATES' : error ? 'API OFFLINE' : 'API CONNECTED · POLLING'}</span><span className="divider"/><span className="camera-tag"><Radio size={14}/> CAM-01 <b>·</b> NORTH GATE</span></div>
      <button className="icon-button" onClick={refresh} title="Refresh dashboard"><RefreshCw size={16}/></button>
    </header>

    <section className="intro"><div><div className="eyebrow"><span/> TRAFFIC MONITORING <span className="slash">/</span> OVERVIEW</div><h1>Vehicle analytics<span>.</span></h1><p>Live crossing events and traffic composition for your camera network.</p></div><div className="date-chip"><span className="status-dot"/> SYSTEM {error ? 'DISCONNECTED' : 'OPERATIONAL'}<small>PROTOTYPE · {metrics.mode?.toUpperCase()}</small></div></section>
    {error && <div className="alert"><CircleHelp size={16}/>{error}</div>}

    <section className="metric-grid">
      <article className="metric-card total-card"><div className="card-label">TOTAL CROSSINGS <span className="card-icon"><CarFront size={16}/></span></div><div className="metric-number">{metrics.total.toLocaleString()}</div><div className="metric-foot"><span className="accent-chip"><Activity size={12}/> ALL TIME</span><span>Recorded events</span></div></article>
      <article className="metric-card"><div className="card-label">INBOUND <span className="direction-icon in"><ArrowDownLeft size={15}/></span></div><div className="metric-number">{(directions.in || 0).toLocaleString()}</div><div className="metric-foot"><span className="mini-line inbound"/><span>Crossed toward site</span></div></article>
      <article className="metric-card"><div className="card-label">OUTBOUND <span className="direction-icon out"><ArrowUpRight size={15}/></span></div><div className="metric-number">{(directions.out || 0).toLocaleString()}</div><div className="metric-foot"><span className="mini-line outbound"/><span>Crossed away from site</span></div></article>
      <article className="metric-card network-card"><div className="card-label">API / LIVE FEED <span className="card-icon"><Wifi size={15}/></span></div><div className="stream-state"><i className={connected || !error ? 'on' : ''}/>{error ? 'Waiting for API' : connected ? 'Live updates connected' : 'API connected · polling'}</div><div className="metric-foot"><span className="camera-tag">RTSP <b>·</b> CAM-01</span></div></article>
    </section>

    <section className="content-grid">
      <article className="panel composition"><div className="panel-heading"><div><div className="eyebrow">CLASSIFICATION</div><h2>Vehicle color</h2></div><span className="panel-tag">{metrics.total} TOTAL</span></div><div className="color-list">{COLORS.map((color) => { const count = palette[color] || 0; return <div className="color-row" key={color}><span className="swatch" style={{'--swatch':colorHex[color]}}/><span className="color-name">{color}</span><div className="bar-track"><i style={{width:`${count / maxColor * 100}%`,background:colorHex[color]}}/></div><b className="color-count">{count}</b></div>; })}</div><div className="panel-note"><span className="note-dot"/>{metrics.mode === 'opencv' ? 'Color is estimated from the vehicle crop using HSV rules; it is not an ML color classifier.' : metrics.mode === 'deepstream' ? 'Color labels come from the active DeepStream event source.' : 'Color values are simulated in demo mode.'}</div></article>
      <article className="panel event-panel"><div className="panel-heading"><div><div className="eyebrow">EVENT LOG</div><h2>Recent crossings</h2></div><span className="live-label"><i/> AUTO-REFRESH</span></div><div className="table-wrap"><table><thead><tr><th>TIME</th><th>VEHICLE</th><th>COLOR</th><th>DIRECTION</th><th>SOURCE</th></tr></thead><tbody>{events.map((event) => <tr key={event.id || event.event_key}><td className="time-cell">{fmt(event.occurred_at)}</td><td><span className="vehicle-cell"><CarFront size={15}/>{event.vehicle_class}</span><small className="vehicle-id">{event.vehicle_id}</small></td><td><span className="color-cell"><i style={{background:colorHex[event.color] || '#888'}}/>{event.color}</span></td><td><span className={`direction-pill ${event.direction}`}>{event.direction === 'in' ? <ArrowDownLeft size={12}/> : <ArrowUpRight size={12}/>} {event.direction}</span></td><td><span className="source-tag">{event.source === 'simulated' ? 'DEMO' : event.source.toUpperCase()}</span></td></tr>)}{events.length === 0 && <tr><td colSpan="5" className="empty">Events will appear here as vehicles cross the line.</td></tr>}</tbody></table></div><div className="table-footer"><span>Showing {events.length} most recent events</span><span className="table-live"><i/> STREAMING</span></div></article>
    </section>
    <footer><span>ROADWATCH <b>·</b> VEHICLE ANALYTICS PROTOTYPE</span><span>{metrics.mode === 'opencv' ? 'OPENCV VIDEO · POSTGRESQL EVENTS' : 'EVENTS PERSISTED IN POSTGRESQL'}</span></footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App/>);
