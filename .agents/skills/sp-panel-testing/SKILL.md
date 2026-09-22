---
name: sp-panel-testing
description: Run browser validation of the independent São Paulo ITBI FastAPI panel with real pipeline artifacts.
---

# SP panel runtime testing

## Setup
- Use `src/gjurema/api/static/index.html` and `src/gjurema/api/app.py` to identify current tabs and API controls; this panel is separate from the national Streamlit dashboard.
- Reuse the project venv, or the shared `/home/ubuntu/repos/gjurema-mvp/.venv` when available. From repo root run `PYTHONPATH=src <venv>/bin/python -m uvicorn gjurema.api.app:app --host 0.0.0.0 --port 8000`.
- Check active pipeline processes and logs before touching artifacts. Never restart a pipeline being used by another session. Wait for complete artifacts, including SP model metadata and metrics; do not use mock data to claim real-data coverage.
- Compare API meta with current artifact metrics. Restart the API after a generation change if needed. Missing artifacts should be explicitly unavailable, not invented data.
- Maximize Chrome with `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz` before recording.

## Devin Secrets Needed
- None for local mode when `GJUREMA_API_TOKEN` is unset. If token protection is enabled, coordinate the intended authentication path; do not expose tokens in logs or screenshots.

## Runtime evidence
- Use native UI controls for both personas, tab selection, portfolio selection, geographic selectors, index pills and composition/ranking dimensions.
- Chart.js depends on cdnjs. Separate CDN/network failures from backend errors. Observe Chart instances only to corroborate visible chart data and finite points, never to mutate UI for tests.
- The normal `/api/` rate limit is120 requests/min/IP; avoid aggressive loops and check for429 during realistic navigation.
- Evolution is annual; the market dashboard exposes only the latest12 monthly buckets. To verify an older ingestion year has no monthly gap, corroborate annual UI/API data against monthly transaction counts from the real parquet. Do not claim those older months were visible in the dashboard.
- Audit initial page load too: a missing favicon can yield404 even if all API calls succeed. Distinguish expected negative-test400 from unexpected failures.
- Missing building history, inventory, builder statistics and CRM data need explicit notices. Check that fallback wording still matches geographic/comparison controls.
