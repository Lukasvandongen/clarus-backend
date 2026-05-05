import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from openai import OpenAI

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except Exception:  # pragma: no cover - firebase-admin is optional until admin logs are enabled.
    firebase_admin = None
    auth = None
    credentials = None

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clarus")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLARUS_MODEL = os.getenv("CLARUS_MODEL", "gpt-5.4-nano")
CLARUS_FALLBACK_MODEL = os.getenv("CLARUS_FALLBACK_MODEL", "gpt-5.4-mini")
CLARUS_MAX_OUTPUT_TOKENS = int(os.getenv("CLARUS_MAX_OUTPUT_TOKENS", "700"))
CLARUS_LOG_PATH = Path(os.getenv("CLARUS_LOG_PATH", "logs/clarus_interactions.jsonl"))
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "luks@degrondvraag.com").strip().lower()
IP_HASH_SALT = os.getenv("CLARUS_IP_HASH_SALT", "")

DEFAULT_ORIGINS = [
    "https://www.degrondvraag.com",
    "https://degrondvraag.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})


PROJECT_CONTEXT = """
degrondvraag.com is an independent essay project about moral, religious, philosophical and existential questions.
It exists to slow down thought: not to deliver quick certainty, but to expose assumptions, clarify terms and help readers think more carefully with the essays.
The site was built and is maintained by its private administrator and author. The administrator deliberately keeps his identity outside the public experience. Do not name, infer or speculate about him.
Clarus is the site's reflective assistant. The name comes from the Latin word for clarity. Clarus is not a database of revelations and not a religious authority. It is a language model guided by pre-written instructions, essay context and project context to help readers formulate clearer questions, distinctions and objections.
Clarus currently has no Qdrant or vector RAG database. It should not pretend to retrieve external sources. It should work from the essay text supplied by the frontend, the conversation and the project context above.
""".strip()


SYSTEM_PROMPT = """
You are Clarus, the reflective assistant of degrondvraag.com.

Primary task:
- Help readers understand, question and refine the essay they are reading.
- Use high academic language, but remain intelligible and precise.
- Respond in the user's language. Use Dutch for Dutch questions and English for English questions. If the frontend provides a language value, follow it.

Knowledge boundaries:
- You may use the supplied essay text, the supplied conversation history and the project context.
- You may use general conceptual vocabulary from philosophy, ethics, theology, literary interpretation and critical theory.
- You must not claim live retrieval, Qdrant search, RAG access or access to a private knowledge base.
- Do not make factual claims about the anonymous administrator beyond the project context.
- Do not identify, infer or speculate about the administrator.

Bible and religion:
- Do not default to biblical interpretation.
- Do not cite the Bible as an authority unless the user explicitly asks for biblical comparison, scriptural context or theological framing.
- If such a comparison is requested, make clear that you are offering a conceptual comparison from general knowledge, not retrieval from a Bible database.

Style:
- Prefer compact, rigorous paragraphs.
- Define key distinctions before giving an answer.
- Ask one clarifying question only when the user's question is too ambiguous to answer responsibly.
- Avoid therapeutic language, motivational phrasing and casual chatbot filler.
- Do not overstate certainty.
- Never pretend to be human. If asked what you are, say you are Clarus, a model guided by pre-written instructions for this site.
- Use Markdown for structure when it helps: short headings, numbered lists, bold key terms and italics for conceptual emphasis.
- Do not use em dashes. Use commas, semicolons or parentheses instead.
""".strip()


ABOUT_CLARUS = {
    "nl": {
        "title": "Over Clarus",
        "body": (
            "Clarus is de reflectieve assistent van degrondvraag.com. De naam verwijst naar helderheid. "
            "Clarus is een taalmodel met vooraf geschreven instructies, essaycontext en projectcontext. "
            "Het systeem gebruikt nu geen Qdrant- of RAG-database en hoort geen externe bronnen te verzinnen. "
            "Gesprekken worden gelogd om fouten, stijl en bruikbaarheid te kunnen beoordelen. Deel daarom geen persoonlijke of gevoelige informatie."
        ),
    },
    "en": {
        "title": "About Clarus",
        "body": (
            "Clarus is the reflective assistant of degrondvraag.com. The name points to clarity. "
            "Clarus is a language model guided by pre-written instructions, essay context and project context. "
            "It currently uses no Qdrant or RAG database and should not invent external retrieval. "
            "Conversations are logged so errors, style and usefulness can be reviewed. Do not share personal or sensitive information."
        ),
    },
}


def strip_html(value: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", value or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_text(value: str, limit: int) -> str:
    value = value or ""
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "..."


def normalize_language(value: str, question: str) -> str:
    if value in {"nl", "en"}:
        return value
    dutch_markers = {"wat", "waarom", "hoe", "essay", "vraag", "bedoelt", "kun", "niet", "wel"}
    words = set(re.findall(r"[a-zA-Z]+", (question or "").lower()))
    return "nl" if words & dutch_markers else "en"


def build_messages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    question = trim_text(data.get("vraag", ""), 1800)
    language = normalize_language(data.get("language", ""), question)
    essay_title = trim_text(data.get("essayTitle", ""), 240)
    essay = trim_text(strip_html(data.get("essay", "")), 12000)
    history = data.get("history", [])

    language_instruction = (
        "Antwoord in het Nederlands." if language == "nl" else "Answer in English."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"{language_instruction}\n\nProject context:\n{PROJECT_CONTEXT}"},
        {
            "role": "system",
            "content": f"Essay title: {essay_title or 'Untitled'}\n\nEssay text supplied by the frontend:\n{essay}",
        },
    ]

    for msg in history[-8:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": trim_text(content, 1400)})

    messages.append({"role": "user", "content": question})
    return messages


def create_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    last_error: Optional[Exception] = None
    for model in [CLARUS_MODEL, CLARUS_FALLBACK_MODEL]:
        if not model:
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=CLARUS_MAX_OUTPUT_TOKENS,
            )
            content = (response.choices[0].message.content or "").strip()
            usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
            return {"answer": content, "model": model, "usage": usage}
        except Exception as exc:  # Try the configured fallback before failing.
            last_error = exc
            logger.warning("Clarus model call failed for %s: %s", model, exc)

    raise last_error or RuntimeError("No model configured.")


def sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_completion(messages: List[Dict[str, str]]):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    last_error: Optional[Exception] = None
    for model in [CLARUS_MODEL, CLARUS_FALLBACK_MODEL]:
        if not model:
            continue
        answer_parts: List[str] = []
        usage: Dict[str, Any] = {}
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=CLARUS_MAX_OUTPUT_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
            yield "model", {"model": model}

            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage.model_dump()
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                token = getattr(delta, "content", None)
                if token:
                    answer_parts.append(token)
                    yield "token", {"token": token}

            answer = "".join(answer_parts).strip()
            return {"answer": answer, "model": model, "usage": usage}
        except Exception as exc:
            last_error = exc
            logger.warning("Clarus streaming call failed for %s: %s", model, exc)
            yield "status", {"message": "Clarus probeert een fallbackmodel."}

    raise last_error or RuntimeError("No model configured.")


def get_ip_hash() -> Optional[str]:
    if not IP_HASH_SALT:
        return None
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() or request.remote_addr or ""
    return hashlib.sha256(f"{IP_HASH_SALT}:{ip}".encode("utf-8")).hexdigest()


def append_log(entry: Dict[str, Any]) -> None:
    CLARUS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLARUS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log_tail(limit: int = 100) -> List[Dict[str, Any]]:
    if not CLARUS_LOG_PATH.exists():
        return []
    lines = CLARUS_LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))


def init_firebase_admin() -> bool:
    if firebase_admin is None:
        return False
    if firebase_admin._apps:
        return True

    try:
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if service_account:
            firebase_admin.initialize_app(credentials.Certificate(json.loads(service_account)))
        else:
            firebase_admin.initialize_app()
        return True
    except Exception as exc:
        logger.warning("Firebase Admin SDK is not configured: %s", exc)
        return False


def require_admin() -> Optional[Any]:
    if not init_firebase_admin():
        return None

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None

    token = header.removeprefix("Bearer ").strip()
    try:
        decoded = auth.verify_id_token(token)
    except Exception as exc:
        logger.warning("Invalid Firebase token for Clarus logs: %s", exc)
        return None

    email = (decoded.get("email") or "").lower()
    if decoded.get("admin") is True or email == ADMIN_EMAIL:
        return decoded
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": CLARUS_MODEL})


@app.route("/clarus/about", methods=["GET"])
def clarus_about():
    language = request.args.get("language", "nl")
    return jsonify(ABOUT_CLARUS.get(language, ABOUT_CLARUS["nl"]))


@app.route("/chat", methods=["POST"])
def clarus_chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("vraag") or "").strip()
    if not question:
        return jsonify({"error": "Geen vraag ontvangen."}), 400

    language = normalize_language(data.get("language", ""), question)
    messages = build_messages(data)
    log_id = str(uuid.uuid4())

    try:
        result = create_completion(messages)
        answer = result["answer"]
        if not answer:
            raise RuntimeError("Model returned an empty answer.")

        entry = {
            "id": log_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "model": result["model"],
            "essayId": data.get("essayId"),
            "essayTitle": trim_text(data.get("essayTitle", ""), 240),
            "question": trim_text(question, 2400),
            "answer": trim_text(answer, 5000),
            "usage": result.get("usage", {}),
            "ipHash": get_ip_hash(),
            "userAgent": trim_text(request.headers.get("User-Agent", ""), 300),
        }
        append_log(entry)
        return jsonify({"antwoord": answer, "logId": log_id, "model": result["model"]})

    except Exception as exc:
        logger.exception("Clarus error")
        append_log({
            "id": log_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "essayId": data.get("essayId"),
            "essayTitle": trim_text(data.get("essayTitle", ""), 240),
            "question": trim_text(question, 2400),
            "error": str(exc),
            "ipHash": get_ip_hash(),
            "userAgent": trim_text(request.headers.get("User-Agent", ""), 300),
        })
        return jsonify({"error": "Er ging iets mis met Clarus."}), 500


@app.route("/chat-stream", methods=["POST"])
def clarus_chat_stream():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("vraag") or "").strip()
    if not question:
        return jsonify({"error": "Geen vraag ontvangen."}), 400

    language = normalize_language(data.get("language", ""), question)
    messages = build_messages(data)
    log_id = str(uuid.uuid4())

    @stream_with_context
    def generate():
        try:
            yield sse("status", {"message": "Clarus heeft de vraag ontvangen."})
            completion_stream = stream_completion(messages)
            while True:
                try:
                    event, payload = next(completion_stream)
                    yield sse(event, payload)
                except StopIteration as done:
                    result = done.value
                    break

            if not result:
                raise RuntimeError("Model returned no result.")

            answer = result.get("answer", "")
            if not answer:
                raise RuntimeError("Model returned an empty answer.")

            append_log({
                "id": log_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "model": result["model"],
                "essayId": data.get("essayId"),
                "essayTitle": trim_text(data.get("essayTitle", ""), 240),
                "question": trim_text(question, 2400),
                "answer": trim_text(answer, 5000),
                "usage": result.get("usage", {}),
                "ipHash": get_ip_hash(),
                "userAgent": trim_text(request.headers.get("User-Agent", ""), 300),
            })
            yield sse("done", {"logId": log_id, "model": result["model"]})
        except Exception as exc:
            logger.exception("Clarus streaming error")
            append_log({
                "id": log_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "essayId": data.get("essayId"),
                "essayTitle": trim_text(data.get("essayTitle", ""), 240),
                "question": trim_text(question, 2400),
                "error": str(exc),
                "ipHash": get_ip_hash(),
                "userAgent": trim_text(request.headers.get("User-Agent", ""), 300),
            })
            yield sse("error", {"error": "Er ging iets mis met Clarus."})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/admin/clarus/logs", methods=["GET"])
def clarus_logs():
    if require_admin() is None:
        return jsonify({"error": "Niet bevoegd."}), 403

    limit = min(max(int(request.args.get("limit", "100")), 1), 500)
    return jsonify({"logs": read_log_tail(limit)})


if __name__ == "__main__":
    app.run(debug=True)
