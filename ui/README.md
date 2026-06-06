# Arabic Complaint Intelligence UI

Static landing page and live demo for the complaint classification API.

## Run locally

Start the API on port 8000, then serve this folder on port 3000:

```bash
# terminal 1 — API
uvicorn main:app --reload

# terminal 2 — UI
cd ui && npm run dev
```

Open http://localhost:3000

Alternatively, without npm:

```bash
cd ui && python3 -m http.server 3000
```
