---
name: test-sp-portfolio-panel
description: Test the São Paulo FastAPI portfolio panel using real artifacts, reversible portfolio edits, and document fixtures.
---

# Runtime and isolation
- Use the checkout and port specified by the task. A baseline may run separately; do not restart or query it unless explicitly requested.
- Reuse existing ITBI artifacts in data/processed and artifacts. Do not launch pipeline runs while a test is reading their outputs.
- Start only if necessary: `PYTHONPATH=src .venv/bin/python -m uvicorn gjurema.api.app:app --host 0.0.0.0 --port <requested-port>`.
- Confirm the effective service environment; protected deployments require credentials. Local development can run without a configured token.
- Maximize Chrome using `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz` before recording.

# Devin Secrets Needed
- None for an explicitly unprotected local instance.
- `GJUREMA_API_TOKEN` if the target deployment enables private portfolio protection.

# Reversible UI tests
- Before PATCH/POST actions, save carteira.json byte-for-byte outside /tmp, record SHA256 and git diff. Never assume HEAD matches pre-test data.
- Create fictional manual and document-based properties through the visible Cadastro tab. Generate PDF, DOCX and OCR PNG containing overlapping and complementary fields; separately test an incomplete TXT and unsupported extension.
- Clear the form before independent documents; also test a sequential upload without clearing to detect stale fields being mistaken for extracted values.
- Verify rental chart after first visiting monthly mode without rent, then adding rent. A fresh annual-to-monthly transition alone can miss selection-state defects.
- Inspect UI request responses and Chart.js datasets without issuing substitute backend mutations. Compare monthly index rates with the generated parquet and distinguish missing values from zero.
- After testing restore only the saved portfolio bytes, verify SHA256 and reload the UI to confirm originals. Preserve unrelated worktree files.

# Evidence and recovery
- Keep audit JSON, fixtures, backup and plan under a persistent directory, not solely /tmp. Runtime restarts may remove /tmp and active recording state.
- On restart, confirm commit unchanged before retaining earlier screenshots as evidence. Clearly report gaps in console/server log history.
- Record recovery/persistence and final restoration separately if the original recording was interrupted. Do not label partial recordings as uninterrupted coverage.
