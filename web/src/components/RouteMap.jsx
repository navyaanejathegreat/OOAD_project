import { MapContainer, TileLayer, Polyline, Marker, Tooltip, CircleMarker } from "react-leaflet";
import L from "leaflet";

// Leaflet's default marker images are not bundled by Vite, so draw our own pins.
const pin = (label, cls) =>
  L.divIcon({ className: "", html: `<div class="pin ${cls}"><span>${label}</span></div>`, iconSize: [30, 30], iconAnchor: [15, 15] });

function colorFor(seg) {
  const t = seg?.time_to_passable;
  if (!t) return "#888";
  if (t.status === "now") return "var(--safe)";
  if (t.status === "in") return "var(--amber)";
  return "var(--stop)";
}

export default function RouteMap({ meta, state, origin, dest, selected, onSelect }) {
  const bySeg = Object.fromEntries((state?.segments || []).map((s) => [s.segment_id, s]));
  const place = Object.fromEntries(meta.places.map((p) => [p.id, p]));
  const rec = state?.advice?.recommended;
  const recCoords = rec?.coords;

  return (
    <MapContainer center={[30.3265, 78.015]} zoom={13} scrollWheelZoom={false} className="map" aria-label="Road map">
      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {meta.roads.map((r) => {
        const seg = r.segment_id ? bySeg[r.segment_id] : null;
        return (
          <Polyline key={r.from + r.to} positions={r.coords}
            pathOptions={{ color: seg ? colorFor(seg) : "#7a8a92", weight: seg ? 9 : 4, opacity: seg ? 0.9 : 0.7 }}
            eventHandlers={r.segment_id ? { click: () => onSelect(r.segment_id) } : undefined}>
            <Tooltip sticky>
              {r.name}
              {seg ? ` · ${seg.depth_cm} cm` : ""}
            </Tooltip>
          </Polyline>
        );
      })}
      {recCoords && (
        <Polyline positions={recCoords} pathOptions={{ color: "#1f4fff", weight: 3, dashArray: "8 8" }} />
      )}
      {selected && bySeg[selected] && (() => {
        const road = meta.roads.find((r) => r.segment_id === selected);
        if (!road) return null;
        const mid = [(road.coords[0][0] + road.coords[1][0]) / 2, (road.coords[0][1] + road.coords[1][1]) / 2];
        return <CircleMarker center={mid} radius={16} pathOptions={{ color: "#1f4fff", weight: 2, fill: false }} />;
      })()}
      {origin && place[origin] && <Marker position={place[origin].coords} icon={pin("A", "a")}><Tooltip>{place[origin].name}</Tooltip></Marker>}
      {dest && place[dest] && <Marker position={place[dest].coords} icon={pin("B", "b")}><Tooltip>{place[dest].name}</Tooltip></Marker>}
    </MapContainer>
  );
}
