const LABEL = { go_now: "Go now", wait: "Wait, then go", no_safe_route: "No safe route" };
const CLASS = { go_now: "go", wait: "wait", no_safe_route: "stop" };

function RouteLine({ title, r }) {
  if (!r) return null;
  return (
    <div className="routeline">
      <span className="eyebrow">{title}</span>
      <span className="mono">
        {r.travel_min} min{r.wait_min > 0 ? ` + ${r.wait_min} min wait` : ""} · arrive {r.eta.slice(11, 16)} · {r.length_km} km
      </span>
    </div>
  );
}

export default function AdvicePanel({ state, loading }) {
  if (!state) return <div className="card advice skeleton">Loading advice…</div>;
  const a = state.advice;
  return (
    <section className={"card advice " + CLASS[a.decision]} aria-live="polite" aria-busy={loading}>
      <div className="advice-head">
        <span className={"pill " + CLASS[a.decision]}>{LABEL[a.decision] || a.decision}</span>
        <span className="muted">Risk: {a.risk}</span>
      </div>
      <p className="message">{a.message}</p>
      <div className="routes">
        <RouteLine title="Go now" r={a.go_now} />
        <RouteLine title="Wait, then go" r={a.wait_then_go} />
      </div>
      {state.naive_unsafe_names.length > 0 && (
        <p className="muted small">
          A water-blind router would send you through: <b>{state.naive_unsafe_names.join(", ")}</b>.
        </p>
      )}
    </section>
  );
}
