# Plan: README.md & .env.example for Word Embedding Chatbot API

## Context

This project is a **dual-server RAG chatbot system** that compares a **proposed hybrid embedding model** (Word2Vec/FastText + BM25 + MMR reranking + RAGAS evaluation) against a **baseline model**.

- **Node.js API** (Hono + Drizzle ORM + SQLite/Turso) — port `9999` — main API gateway with OpenAPI docs at `/reference`
- **Python ML Server** (FastAPI + gensim + LangChain) — port `8888` — document training, hybrid semantic search, GPT answer generation

The Node API forwards PDF training & query requests to the Python server.

---

## Deliverables

1. **`.env.example`** — placeholder template for all env vars (Node + Python)
2. **`README.md`** — full project documentation

---

## .env.example Structure

A single file covering both servers, with placeholder values and inline comments:

```
# ─── NODEJS SERVER ENV ───
NODE_ENV=development
PORT=9999
LOG_LEVEL=debug
DATABASE_URL=file:dev.db
DATABASE_AUTH_TOKEN=               # required in production (Turso)
ACCESS_TOKEN_SECRET=               # must be > 32 characters
ACCESS_TOKEN_SECRET_PUBLIC=        # must be > 32 characters
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
PINECONE_API_KEY=
PYTHON_SERVER_URL=http://localhost:8888
OPEN_AI_API_KEY=

# ─── PYTHON SERVER ENV ───
ALLOWED_ORIGINS=*
API_PASSWORD=
OPENAI_API_KEY=
PORT_PY=8888
```

---

## README.md Structure

### 1. Title & Badges

- Project name, one-line description
- Badges: MIT License, Node.js, Python, Hono, FastAPI

### 2. Table of Contents

Anchor-linked TOC to every section below.

### 3. Overview

- What the project does: a RAG chatbot that ingests PDFs, trains word-embedding models, and answers questions using hybrid semantic search + GPT generation
- Key differentiator: side-by-side comparison of a **proposed hybrid model** (FastText/Word2Vec + BM25 + MMR + RAGAS) vs a **baseline model**
- Indonesian-language NLP (Sastrawi stemmer, Indonesian stopwords)

### 4. Architecture

Mermaid diagram showing data flow:

```mermaid
flowchart LR
    Client[Client / Frontend] --> NodeAPI[Node.js API - Hono - port 9999]
    NodeAPI --> SQLite[(SQLite / Turso)]
    NodeAPI --> Redis[(Upstash Redis - rate limit)]
    NodeAPI -->|train / query> PyAPI[Python ML Server - FastAPI - port 8888]
    PyAPI --> Gensim[Word2Vec / FastText models]
    PyAPI --> BM25[BM25 Okapi]
    PyAPI -->|optional> OpenAI[OpenAI GPT]
    NodeAPI -->|optional embedding> Pinecone[(Pinecone Vector DB)]
    NodeAPI -->|PDF storage| Cloudinary[(Cloudinary)]
```

### 5. Tech Stack

Two-column table:

| Node.js API                | Python ML Server            |
| -------------------------- | --------------------------- |
| Hono                       | FastAPI                     |
| Drizzle ORM                | gensim (Word2Vec, FastText) |
| SQLite / Turso (libSQL)    | rank-bm25                   |
| Zod / @hono/zod-openapi    | LangChain                   |
| @scalar/hono-api-reference | scikit-learn                |
| argon2 (password hashing)  | NLTK + Sastrawi             |
| hono-pino (logging)        | PyMuPDF / pdf-parse         |
| hono-rate-limiter          | OpenAI Python SDK           |
| Upstash Redis              | numpy / scipy / pandas      |
| Pinecone client            | uvicorn                     |
| OpenAI SDK                 | python-dotenv               |

### 6. Minimum Requirements

- **Node.js** >= 20 (ESM, uses `import.meta`)
- **npm** >= 10
- **Python** 3.11.12 (pinned in `.python-version`)
- **git**
- OS: Linux / macOS / Windows (WSL recommended)

### 7. Prerequisites — Third-Party Services

- **Turso / libSQL** — SQLite database (local `file:dev.db` for dev, Turso cloud for prod)
- **Upstash Redis** — rate limiting
- **Pinecone** — optional vector DB for the `pinecone` embedding mode
- **OpenAI API key** — GPT answer generation
- **Cloudinary** — optional PDF storage

### 8. Environment Variables

Full table with columns: Variable | Server | Description | Required | Default

Covers all vars from `src/env.ts` Zod schema + Python vars from `.env`.

### 9. Installation & Setup — Node.js API

```bash
# 1. Clone
git clone <repo-url>
cd word-embeding-chatbot-api

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# edit .env with your values

# 4. Run database migrations
npx drizzle-kit migrate

# 5. Start dev server
npm run dev
# → http://localhost:9999
```

### 10. Installation & Setup — Python ML Server

> **Recommendation:** Use a Python virtual environment (`venv`) to isolate dependencies and avoid system-wide conflicts.

```bash
# 1. Create virtual environment
python3.11 -m venv venv

# 2. Activate it
#    Linux / macOS:
source venv/bin/activate
#    Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLTK data (happens automatically on first run, or manually)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 6. Start the ML server
uvicorn api:app --host 0.0.0.0 --port 8888 --reload
# → http://localhost:8888
```

### 11. Available Scripts / Commands

| Command                                | Server | Description                        |
| -------------------------------------- | ------ | ---------------------------------- |
| `npm run dev`                          | Node   | Start dev server with tsx watch    |
| `npm run build`                        | Node   | Compile TypeScript to `dist/`      |
| `npm start`                            | Node   | Run compiled production build      |
| `npm run lint`                         | Node   | Run ESLint                         |
| `npm run lint:fix`                     | Node   | Auto-fix lint issues               |
| `npm run typecheck`                    | Node   | Type-check without emitting        |
| `npm test`                             | Node   | Run Vitest test suite              |
| `npx drizzle-kit migrate`              | Node   | Apply DB migrations                |
| `npx drizzle-kit generate`             | Node   | Generate new migration from schema |
| `uvicorn api:app --port 8888 --reload` | Python | Start ML server in dev             |

### 12. API Reference

- **Interactive docs (Scalar):** `http://localhost:9999/reference`
- **OpenAPI spec (JSON):** `http://localhost:9999/doc`
- **Python docs (Swagger):** `http://localhost:8888/docs`

Key Node.js endpoints grouped by resource:

- `Auth` — `POST /auth/register`, `POST /auth/login`
- `Chatbots` — `GET/POST /chatbots`, `GET/PATCH/DELETE /chatbots/:id`
- `Conversations` — `GET/POST /conversations`, ...
- `Messages` — `GET/POST /messages`, ...
- `Contacts` — `GET/POST /contacts`, ...
- `Tasks` — `GET/POST /tasks`, ...

Key Python ML endpoints:

- `POST /train/proposed-model` | `POST /train/baseline-model`
- `POST /query/proposed-model` | `POST /query/baseline-model`
- `GET /models/{userId}/{chatbotId}/proposed` | `/baseline`
- `DELETE /models/{userId}/{chatbotId}/{pdfTitle}/proposed` | `/baseline`
- `GET /health/proposed`

### 13. Project Structure

```
word-embeding-chatbot-api/
├── api.py                    # Python FastAPI ML server
├── requirements.txt          # Python dependencies
├── .python-version           # Python 3.11.12
├── src/
│   ├── app.ts                # Hono app + route registration
│   ├── index.ts              # Server entry point
│   ├── env.ts                # Zod-validated env config
│   ├── db/
│   │   ├── index.ts          # Drizzle client (libSQL)
│   │   ├── schema.ts         # DB schema + Zod schemas
│   │   ├── pinecone.ts       # Pinecone vector integration
│   │   ├── cloudinary.ts     # Cloudinary integration
│   │   ├── redish.ts         # Upstash Redis client
│   │   └── migrations/       # SQL migrations
│   ├── lib/
│   │   ├── configure-open-api.ts
│   │   ├── create-app.ts
│   │   ├── token.ts          # JWT helpers
│   │   ├── hashing.ts        # argon2
│   │   ├── send-training-request-*.ts   # Python server proxy
│   │   ├── send-query-request-*.ts      # Python server proxy
│   │   └── ...
│   ├── middlewares/
│   │   ├── auth.middleware.ts
│   │   ├── body-limit.middleware.ts
│   │   ├── limiter.middleware.ts
│   │   └── pino-logger.ts
│   └── routes/
│       ├── auth/
│       ├── chatbots/
│       ├── contacts/
│       ├── conversations/
│       ├── messages/
│       └── tasks/
├── drizzle.config.ts
├── package.json
└── tsconfig.json
```

### 14. Usage Workflow

Step-by-step:

1. Register a user → `POST /auth/register`
2. Login → `POST /auth/login` (returns JWT)
3. Create a chatbot with a PDF → `POST /chatbots` (triggers training on Python server)
4. Create a conversation → `POST /conversations`
5. Send a message / query → `POST /messages` (Node forwards to Python `/query/proposed-model` or `/baseline-model`)
6. Compare proposed vs baseline model responses (stored in `modelResponses` table)

### 15. Security Note

- ⚠️ Never commit real `.env` files. `.env` is in `.gitignore`.
- Use `.env.example` as a template.
- Rotate any keys currently exposed in version control.
- Set strong `ACCESS_TOKEN_SECRET` values (> 32 chars).

### 16. License & Contributing

- MIT License — © KuraoHikari
- Contributing guidelines (fork → branch → PR)

---

## Implementation Notes

- All commands use **npm** (per `package.json` scripts).
- Python section strongly recommends **venv**.
- README written in **English**.
- Both servers get **full setup instructions**.
- `.env.example` created with **placeholder values only** (no real secrets).
