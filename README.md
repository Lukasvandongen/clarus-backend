# Clarus Backend

Clarus is the reflective assistant for degrondvraag.com. It no longer depends on Qdrant or a Bible retrieval database. The backend sends the current essay text, project context and strict system instructions to a low-cost OpenAI model.

## Environment

Set these variables on Render:

```env
OPENAI_API_KEY=...
CLARUS_MODEL=gpt-5.4-nano
CLARUS_FALLBACK_MODEL=gpt-5.4-mini
CLARUS_MAX_OUTPUT_TOKENS=700
ADMIN_EMAIL=luks@degrondvraag.com
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
CLARUS_LOG_COLLECTION=clarusLogs
CLARUS_LOG_PATH=logs/clarus_interactions.jsonl
CLARUS_IP_HASH_SALT=choose-a-long-random-secret
CORS_ORIGINS=https://www.degrondvraag.com,https://degrondvraag.com,http://localhost:5173
```

`FIREBASE_SERVICE_ACCOUNT_JSON` is needed for persistent Firestore logs and the admin log endpoint. Without it, `/chat` still works and writes the local JSONL fallback, but Render's filesystem should not be treated as durable storage.

## Routes

- `GET /health` checks whether the service is alive.
- `GET /clarus/about?language=nl` returns the public Clarus explanation.
- `POST /chat` answers a Clarus question and writes one JSONL log entry.
- `POST /chat-stream` streams a Clarus answer as server-sent events and writes one JSONL log entry.
- `GET /admin/clarus/logs` returns recent logs for Firebase admins.

## Logs

Interactions are written to Firestore collection `clarusLogs` when Firebase Admin is configured. The backend also writes a local JSONL fallback. The log entry stores the question, answer, model, usage, essay id, essay title, language, user agent, status and an optional salted IP hash. Do not enable the IP hash unless you have a clear reason to keep it.

Clarus has a strict scope guard. It should answer only about essays, morality, religion as a concept, philosophy, existential questions, argument analysis and relevant criticism of the site. Obvious coding or general assistant requests are refused before a model call is made.
