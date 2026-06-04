## Complaint Analyst UI

This folder contains a lightweight, LibreChat-inspired frontend for the
`text-complaint-api` backend.

## Getting Started

1) Install dependencies:

```bash
npm install
```

2) Configure API URL:

```bash
cp .env.local.example .env.local
```

Default value:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

3) Run the dev server:

```bash
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Features

- Chat-style complaint input flow
- Arabic complaint support (RTL input)
- Displays `sentiment`, `topic`, `intent`, and final `action`
- Toggle between `/predict` and `/explain-classification`
- Shows explanation metadata and error hints for LLM fallback cases

## Backend requirement

Run the backend API from the repository root:

```bash
uvicorn main:app --reload
```
