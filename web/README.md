# JalMarg web app (React)

React + Vite + Leaflet front end for the JalMarg FastAPI server. Pick a vehicle, drag or play the
storm timeline, and see the wait-or-detour advice, the coloured road map and the depth forecast.

## Run it

Easiest, from the project root (starts the API and the web app together):

```powershell
.\run_web.ps1
```

Then open http://localhost:5173.

Or in two terminals:

```powershell
# terminal 1, project root: the API on port 8000
python -m uvicorn jalmarg.api:app --port 8000

# terminal 2: the React app on port 5173
cd web
npm install      # first time only
npm run dev
```

The React dev server forwards `/api/*` to the API on port 8000, so there is nothing to configure.

## Files

| File | What it does |
|---|---|
| `src/App.jsx` | State and controls: vehicle, time slider, replay, from/to, forecaster |
| `src/api.js` | Calls `/demo/meta` and `/demo/state` |
| `src/components/RouteMap.jsx` | Leaflet map, segments coloured for your vehicle, recommended route |
| `src/components/AdvicePanel.jsx` | Go / wait / no-safe-route banner and route times |
| `src/components/SegmentTable.jsx` | Depth and time-to-passable per segment |
| `src/components/ForecastChart.jsx` | Depth history, median, 90th percentile, band, vehicle limit |

The map tiles come from OpenStreetMap, so the map background needs internet.
