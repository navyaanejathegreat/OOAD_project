import { useEffect, useMemo, useRef, useState } from "react";
import { fetchMeta, fetchState } from "./api.js";
import RouteMap from "./components/RouteMap.jsx";
import AdvicePanel from "./components/AdvicePanel.jsx";
import SegmentTable from "./components/SegmentTable.jsx";
import ForecastChart from "./components/ForecastChart.jsx";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [vehicle, setVehicle] = useState("two_wheeler");
  const [minute, setMinute] = useState(40);
  const [model, setModel] = useState("hybrid");
  const [origin, setOrigin] = useState("home");
  const [dest, setDest] = useState("office");
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState("underpass");
  const [playing, setPlaying] = useState(false);
  const reqId = useRef(0);

  useEffect(() => {
    fetchMeta().then(setMeta).catch((e) => setError(e.message));
  }, []);

  // Refetch whenever an input changes; ignore answers from outdated requests.
  useEffect(() => {
    if (!meta) return;
    const id = ++reqId.current;
    setLoading(true);
    const t = setTimeout(() => {
      fetchState({ vehicle, minute, model, origin, dest })
        .then((s) => {
          if (id === reqId.current) {
            setState(s);
            setError(null);
          }
        })
        .catch((e) => id === reqId.current && setError(e.message))
        .finally(() => id === reqId.current && setLoading(false));
    }, 120);
    return () => clearTimeout(t);
  }, [meta, vehicle, minute, model, origin, dest]);

  // Replay the storm: advance 2 minutes per tick.
  useEffect(() => {
    if (!playing || !meta) return;
    const t = setInterval(() => {
      setMinute((m) => {
        if (m >= meta.max_minute) {
          setPlaying(false);
          return m;
        }
        return Math.min(m + 2, meta.max_minute);
      });
    }, 700);
    return () => clearInterval(t);
  }, [playing, meta]);

  const veh = useMemo(() => meta?.vehicles.find((v) => v.key === vehicle), [meta, vehicle]);

  if (error && !meta) {
    return (
      <main className="shell">
        <h1>JalMarg</h1>
        <div className="banner error" role="alert">
          <b>Cannot reach the JalMarg API.</b> Start it in another terminal with{" "}
          <code>python -m uvicorn jalmarg.api:app --port 8000</code>, then reload this page.
          <div className="muted">{error}</div>
        </div>
      </main>
    );
  }
  if (!meta) return <main className="shell"><p className="muted">Loading…</p></main>;

  const clock = state ? state.time.slice(11, 16) : "--:--";

  return (
    <main className="shell">
      <header className="top">
        <div>
          <h1>JalMarg <span className="deva">जलमार्ग</span></h1>
          <p className="muted">Wait or detour? A recorded storm replayed as a live stream. Roads are illustrative.</p>
        </div>
        <div className="clock" aria-live="polite">
          <span className="eyebrow">Time</span>
          <b>{clock}</b>
          <span className={"rain " + (state?.raining ? "on" : "")}>
            {state?.raining ? `Raining ${meta.rain_mm_h} mm/h` : "Rain stopped"}
          </span>
        </div>
      </header>

      <section className="controls" aria-label="Controls">
        <div className="field">
          <span className="eyebrow">Your vehicle</span>
          <div className="seg" role="group" aria-label="Vehicle class">
            {meta.vehicles.map((v) => (
              <button key={v.key} type="button" aria-pressed={v.key === vehicle} onClick={() => setVehicle(v.key)}>
                {v.label}
              </button>
            ))}
          </div>
        </div>

        <div className="field grow">
          <label htmlFor="minute" className="eyebrow">Minutes since rain started: <b className="mono">{minute}</b></label>
          <div className="row">
            <button type="button" className="play" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause replay" : "Play replay"}>
              {playing ? "Pause" : "Play"}
            </button>
            <input id="minute" type="range" min="0" max={meta.max_minute} value={minute}
              onChange={(e) => { setPlaying(false); setMinute(+e.target.value); }} />
          </div>
        </div>

        <div className="field">
          <label htmlFor="origin" className="eyebrow">From</label>
          <select id="origin" value={origin} onChange={(e) => setOrigin(e.target.value)}>
            {meta.places.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="dest" className="eyebrow">To</label>
          <select id="dest" value={dest} onChange={(e) => setDest(e.target.value)}>
            {meta.places.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="model" className="eyebrow">Forecaster</label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="hybrid">Hybrid (physics + GRU)</option>
            <option value="physics">Physics only</option>
          </select>
        </div>
      </section>

      {error && <div className="banner error" role="alert">{error}</div>}

      <AdvicePanel state={state} loading={loading} />

      <div className="grid">
        <div className="card">
          <RouteMap meta={meta} state={state} origin={origin} dest={dest} selected={selected} onSelect={setSelected} />
          <ul className="legend" aria-label="Map legend">
            <li><i style={{ background: "var(--safe)" }} /> Passable now for you</li>
            <li><i style={{ background: "var(--amber)" }} /> Opens within the hour</li>
            <li><i style={{ background: "var(--stop)" }} /> Not expected / unknown</li>
            <li><i className="dash" /> Recommended route</li>
          </ul>
        </div>
        <div className="card">
          <SegmentTable state={state} selected={selected} onSelect={setSelected} />
        </div>
      </div>

      <div className="card">
        <ForecastChart state={state} selected={selected} limit={veh?.threshold_cm} vehicleLabel={veh?.label} />
      </div>

      <div className="card">
        <h2>Alerts (municipal view)</h2>
        {state?.alerts?.length ? (
          <ul className="alerts">
            {state.alerts.map((a, i) => (
              <li key={i}><span className="mono">{a.time.slice(11, 16)}</span> {a.message}</li>
            ))}
          </ul>
        ) : <p className="muted">No alerts yet at this point in the storm.</p>}
      </div>

      <footer className="muted">{meta.disclaimer}</footer>
    </main>
  );
}
