// SVG forecast chart: tracked truth (history), q10-q90 band, median, q90 used for decisions, and the vehicle limit.
const W = 880, H = 320, L = 56, R = 20, T = 16, B = 38;

export default function ForecastChart({ state, selected, limit, vehicleLabel }) {
  const seg = state?.segments.find((s) => s.segment_id === selected);
  if (!seg) return <p className="muted">Select a segment to see its forecast.</p>;
  const hist = seg.history || [];
  const fc = seg.forecast || [];
  const now = state.minute;

  const xs = [...hist.map((p) => p.minute), ...fc.map((p) => p.minute)];
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ymaxRaw = Math.max(limit || 0, ...hist.map((p) => p.true_cm), ...fc.map((p) => p.q90));
  const ymax = Math.max(10, Math.ceil((ymaxRaw + 2) / 5) * 5);
  const x = (m) => L + ((m - x0) / (x1 - x0 || 1)) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  const path = (pts, key) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.minute).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(" ");

  const band = fc.length
    ? [...fc.map((p) => `${x(p.minute)},${y(p.q90)}`), ...[...fc].reverse().map((p) => `${x(p.minute)},${y(p.q10)}`)].join(" ")
    : "";

  const yticks = [];
  for (let v = 0; v <= ymax; v += ymax > 40 ? 10 : 5) yticks.push(v);
  const xticks = [];
  for (let m = Math.ceil(x0 / 15) * 15; m <= x1; m += 15) xticks.push(m);

  const futureTrue = fc.filter((p) => p.true_cm != null);

  return (
    <>
      <div className="chart-head">
        <h2>Forecast: {seg.name}</h2>
        <span className="muted small">{seg.time_to_passable.text}</span>
      </div>
      <div className="table-box">
        <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
          aria-label={`Depth history and forecast for ${seg.name}, with the ${vehicleLabel} limit of ${limit} centimetres`}>
          {yticks.map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" />
              <text x={L - 8} y={y(v) + 4} textAnchor="end" className="tick">{v} cm</text>
            </g>
          ))}
          {xticks.map((m) => (
            <text key={m} x={x(m)} y={H - B + 20} textAnchor="middle" className="tick">{m - now === 0 ? "now" : (m - now > 0 ? "+" : "") + (m - now) + " min"}</text>
          ))}
          {band && <polygon points={band} className="band" />}
          <line x1={x(now)} x2={x(now)} y1={T} y2={H - B} className="nowline" />
          {fc.length > 0 && <path d={path(fc, "q90")} className="q90" />}
          {fc.length > 0 && <path d={path(fc, "q50")} className="q50" />}
          {futureTrue.length > 0 && <path d={path(futureTrue.map((p) => ({ minute: p.minute, v: p.true_cm })), "v")} className="truth future" />}
          <path d={path(hist.map((p) => ({ minute: p.minute, v: p.true_cm })), "v")} className="truth" />
          {limit != null && (
            <>
              <line x1={L} x2={W - R} y1={y(limit)} y2={y(limit)} className="limit" />
              <text x={W - R - 4} y={y(limit) - 7} textAnchor="end" className="limit-label">{vehicleLabel} limit {limit} cm</text>
            </>
          )}
        </svg>
      </div>
      <ul className="legend inline" aria-label="Chart legend">
        <li><i className="l truth" /> True depth</li>
        <li><i className="l q50" /> Median forecast</li>
        <li><i className="l q90" /> 90th percentile (used for decisions)</li>
        <li><i className="l bandkey" /> 10–90% band</li>
      </ul>
    </>
  );
}
