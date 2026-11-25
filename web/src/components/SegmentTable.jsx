const STATE_CLASS = { Dry: "dry", Wet: "wet", Rising: "rising", Receding: "receding", Unknown: "unknown" };

export default function SegmentTable({ state, selected, onSelect }) {
  if (!state) return <p className="muted">Loading segments…</p>;
  return (
    <>
      <h2>Road segments</h2>
      <div className="table-box">
        <table>
          <thead>
            <tr>
              <th>Segment</th><th>State</th><th className="num">Depth</th><th>For you</th>
            </tr>
          </thead>
          <tbody>
            {state.segments.map((s) => (
              <tr key={s.segment_id} className={s.segment_id === selected ? "sel" : ""}
                  onClick={() => onSelect(s.segment_id)} tabIndex={0}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(s.segment_id)}>
                <td>{s.name}</td>
                <td><span className={"state " + STATE_CLASS[s.state]}>{s.state}</span></td>
                <td className="num">{s.depth_cm} ± {s.sigma_cm} cm</td>
                <td>{s.time_to_passable.text.replace(` for ${state.vehicle.label}`, "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">Click a row to see its forecast below.</p>
    </>
  );
}
