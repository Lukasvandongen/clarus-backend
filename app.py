import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from openai import OpenAI
import qdrant_client
from qdrant_client.http import models as qmodels

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bijbel")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = qdrant_client.QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

# ---------- Bible book normalization (NL) ----------
# compact alias map; breid uit indien nodig
ALIASES = {
    "gen": "Genesis", "genesis": "Genesis",
    "exo": "Exodus", "exodus": "Exodus",
    "lev": "Leviticus", "leviticus": "Leviticus",
    "num": "Numeri", "numeri": "Numeri",
    "deut": "Deuteronomium", "deuteronomium": "Deuteronomium",
    "jos": "Jozua", "jozua": "Jozua",
    "rech": "Richteren", "richteren": "Richteren", "rechteren": "Richteren",
    "rut": "Ruth", "ruth": "Ruth",
    "1sam": "1 Samuël", "1 sam": "1 Samuël", "1 samuel": "1 Samuël",
    "2sam": "2 Samuël", "2 sam": "2 Samuël", "2 samuel": "2 Samuël",
    "1kon": "1 Koningen", "1 koning": "1 Koningen", "1 koningen": "1 Koningen",
    "2kon": "2 Koningen", "2 koning": "2 Koningen", "2 koningen": "2 Koningen",
    "1kron": "1 Kronieken", "1 kron": "1 Kronieken",
    "2kron": "2 Kronieken", "2 kron": "2 Kronieken",
    "ezra": "Ezra", "neh": "Nehemia", "nehemia": "Nehemia",
    "est": "Ester", "ester": "Ester",
    "job": "Job", "ps": "Psalmen", "psalm": "Psalmen", "psalmen": "Psalmen",
    "spr": "Spreuken", "spreuken": "Spreuken",
    "pred": "Prediker", "prediker": "Prediker",
    "hooglied": "Hooglied",
    "jes": "Jesaja", "jesaja": "Jesaja",
    "jer": "Jeremia", "jeremia": "Jeremia",
    "klaagliederen": "Klaagliederen",
    "ez": "Ezechiël", "ezechiel": "Ezechiël",
    "dan": "Daniël", "daniel": "Daniël",
    "hos": "Hosea", "hosea": "Hosea",
    "joel": "Joël", "jool": "Joël", "joël": "Joël",
    "amos": "Amos", "obadja": "Obadja", "jona": "Jona",
    "micha": "Micha", "nahum": "Nahum", "hab": "Habakuk", "habakuk": "Habakuk",
    "zef": "Zefanja", "zefanja": "Zefanja", "haggaï": "Haggai", "haggai": "Haggai",
    "zach": "Zacharia", "zacharia": "Zacharia",
    "mal": "Maleachi", "maleachi": "Maleachi",
    # NT
    "mat": "Mattheüs", "mattheus": "Mattheüs", "matheus": "Mattheüs", "mt": "Mattheüs",
    "mark": "Markus", "marcus": "Markus", "mr": "Markus",
    "luc": "Lukas", "lukas": "Lukas", "lk": "Lukas",
    "joh": "Johannes", "johannes": "Johannes", "jn": "Johannes",
    "hand": "Handelingen", "handelingen": "Handelingen",
    "rom": "Romeinen", "romeinen": "Romeinen",
    "1kor": "1 Korinthe", "1 kor": "1 Korinthe", "1 korinthe": "1 Korinthe", "1 korintiers": "1 Korinthe", "1 korintiërs": "1 Korinthe",
    "2kor": "2 Korinthe", "2 kor": "2 Korinthe", "2 korinthe": "2 Korinthe", "2 korintiërs": "2 Korinthe",
    "gal": "Galaten", "ef": "Efeze", "efeze": "Efeze",
    "fil": "Filippenzen", "filippenzen": "Filippenzen",
    "kol": "Kolossenzen", "kolossenzen": "Kolossenzen",
    "1thess": "1 Thessalonicenzen", "2thess": "2 Thessalonicenzen",
    "1tim": "1 Timotheüs", "2tim": "2 Timotheüs",
    "tit": "Titus", "filemon": "Filemon",
    "heb": "Hebreeën", "hebreeen": "Hebreeën",
    "jak": "Jakobus", "1petr": "1 Petrus", "2petr": "2 Petrus",
    "1joh": "1 Johannes", "2joh": "2 Johannes", "3joh": "3 Johannes",
    "jud": "Judas", "openb": "Openbaring", "openbaring": "Openbaring",
}

BOOK_CANON = set(ALIASES.values()) | {
    "Genesis","Exodus","Leviticus","Numeri","Deuteronomium","Jozua","Richteren","Ruth","1 Samuël","2 Samuël","1 Koningen","2 Koningen","1 Kronieken","2 Kronieken","Ezra","Nehemia","Ester","Job","Psalmen","Spreuken","Prediker","Hooglied","Jesaja","Jeremia","Klaagliederen","Ezechiël","Daniël","Hosea","Joël","Amos","Obadja","Jona","Micha","Nahum","Habakuk","Zefanja","Haggai","Zacharia","Maleachi","Mattheüs","Markus","Lukas","Johannes","Handelingen","Romeinen","1 Korinthe","2 Korinthe","Galaten","Efeze","Filippenzen","Kolossenzen","1 Thessalonicenzen","2 Thessalonicenzen","1 Timotheüs","2 Timotheüs","Titus","Filemon","Hebreeën","Jakobus","1 Petrus","2 Petrus","1 Johannes","2 Johannes","3 Johannes","Judas","Openbaring"
}

REF_RE = re.compile(r"^\s*([1-3]?\s?[A-Za-zÀ-ſ.]+)\s+(\d+)(?::(\d+)(?:-(\d+))?)?\s*$")


def normalize_book(name: str) -> str:
    key = name.strip().lower().replace(".", "").replace("  ", " ")
    key = key.replace("matth", "mat")  # veelgemaakte variant
    return ALIASES.get(key, name.strip().title())


def parse_reference(text: str) -> Optional[Dict[str, Any]]:
    """Parse 'Mattheus 20:1-15' etc. Return dict or None."""
    m = REF_RE.match(text.replace("Mattheus", "Mattheüs"))
    if not m:
        return None
    raw_book, chapter, v1, v2 = m.groups()
    boek = normalize_book(raw_book)
    if boek not in BOOK_CANON:
        return None
    ref = {
        "boek": boek,
        "hoofdstuk": int(chapter),
    }
    if v1:
        ref["vers_start"] = int(v1)
    if v2:
        ref["vers_einde"] = int(v2)
    return ref


# ---------- Qdrant search ----------

def search_qdrant(query: str, k: int = 5, ref: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search Qdrant using embeddings + optional structured filter on boek/hoofdstuk/vers."""
    # Build embedding for vector search
    emb = client.embeddings.create(model="text-embedding-3-small", input=query)
    vector = emb.data[0].embedding

    q_filter = None
    if ref:
        must: List[Any] = []
        if "boek" in ref:
            must.append(qmodels.FieldCondition(key="boek", match=qmodels.MatchValue(value=ref["boek"])))
        if "hoofdstuk" in ref:
            must.append(qmodels.FieldCondition(key="hoofdstuk", match=qmodels.MatchValue(value=ref["hoofdstuk"])))
        # NB: vaak is elk punt 1 vers in de payload; meestal filter je op boek/hoofdstuk en laat je vector de verzen vinden
        if must:
            q_filter = qmodels.Filter(must=must)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=k,
        with_payload=True,
        score_threshold=None,
        query_filter=q_filter,
    )

    results = []
    for h in hits:
        p = h.payload or {}
        if "tekst" not in p:
            continue
        results.append({
            "tekst": p.get("tekst", ""),
            "boek": p.get("boek", ""),
            "hoofdstuk": p.get("hoofdstuk", 0),
            "vers": p.get("vers", ""),
            "score": getattr(h, "score", None),
        })
    return results


# ---------- Flask ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://www.degrondvraag.com",
    "https://degrondvraag.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]}})

# ---------- System prompt (tool-driven) ----------
SYSTEM_PROMPT = (
    """
Jij bent **Clarus** (Latijn: 'duidelijkheid'), de bedachtzame en bondige chat-hulp van degrondvraag.com.

Context & grenzen:
- Baseer je antwoorden **uitsluitend** op (a) de meegegeven essay-tekst en (b) Bijbelverzen die via de tool *search_bible* worden aangeleverd.
- Geen externe kennis of verzinsels. Citeer verzen **letterlijk** en geef altijd de bron (Boek Hoofdstuk:Vers-range).
- Geef compacte, natuurlijke alinea's; gebruik opsommingen alleen als dat functioneel is.
- Zeg nooit dat je een AI bent. Begroet alleen bij het begin van de sessie.

Retrieval-regels (belangrijk):
- Als de vraag Bijbelcontext vereist of refereert aan een passage, roep dan de tool `search_bible` aan met een zo specifiek mogelijke **referentie** (bijv. "Mattheüs 20:1-16").
- Is de gebruiker vaag (bijv. "Mattheus 20" of "Mattheüs 20"), kies dan het **hele hoofdstuk** of vraag om verfijning van de verzen wanneer nodig.
- Let op Nederlandse diakritiek en canonieke namen (bijv. *Mattheüs*, *1 Korinthe*). Normaliseer waar nodig.
- Als *search_bible* niets teruggeeft, vraag om een preciezere referentie of alternatieve formulering.

Antwoordvorm:
1) Als je verzen hebt: eerst **citeren** (letterlijk, met referentie), daarna pas een korte toelichting **alleen als daarom gevraagd wordt**.
2) Als je verzen nodig hebt maar ze ontbreken: vraag expliciet om de referentie of voer `search_bible` uit met je beste gok.
3) Geen samenvattingen of analyses van het hele essay tenzij expliciet gevraagd.
"""
).strip()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_bible",
            "description": "Zoek in Qdrant naar Bijbelverzen op basis van een referentie of zoekterm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Vrije zoekterm of referentie, bijv. 'Mattheüs 20:1-16' of een trefwoord."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]


# ---------- Tool dispatcher ----------

def tool_search_bible(args_json: str) -> str:
    args = json.loads(args_json or "{}")
    raw_query: str = args.get("query", "").strip()
    top_k: int = int(args.get("top_k", 5))

    # probeer referentie parsing
    ref = parse_reference(raw_query)
    # fallback normalisatie (bij "Mattheus 20" zonder dubbelpunt)
    if not ref:
        m = re.match(r"^\s*([1-3]?\s?[A-Za-z\u00c0-\u017f.]+)\s+(\d+)\s*$", raw_query)
        if m:
            ref = {"boek": normalize_book(m.group(1)), "hoofdstuk": int(m.group(2))}

    logging.info(f"[search_bible] query='{raw_query}', ref={ref}")
    results = search_qdrant(raw_query, k=top_k, ref=ref)

    # format compact for the model
    return json.dumps({
        "query": raw_query,
        "ref": ref,
        "hits": results,
    }, ensure_ascii=False)


# ---------- Chat route ----------
@app.route("/chat", methods=["POST"])
def clarus():
    data = request.get_json(force=True, silent=True) or {}
    essay = data.get("essay", "")
    vraag = data.get("vraag", "")
    history: List[Dict[str, str]] = data.get("history", [])

    if not vraag:
        return jsonify({"antwoord": "Geen vraag ontvangen"}), 400

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Essay-context (alleen gebruiken als bron):\n\n{essay}"},
    ]

    for msg in history:
        if msg.get("role") in {"user", "assistant"} and isinstance(msg.get("content"), str):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": vraag})

    try:
        # loop to handle tool calls (max 3 rounds)
        for _ in range(3):
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                # execute tools sequentially
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == "search_bible":
                        tool_payload = tool_search_bible(tool_call.function.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_payload,
                        })
                # continue the loop to let the model consume tool outputs
                continue

            # no tool calls -> we have a final answer
            final_answer = (msg.content or "").strip()
            logging.info(f"[Clarus antwoord] {final_answer[:400]}")
            return jsonify({"antwoord": final_answer})

        # if loop exhausted
        return jsonify({"error": "Teveel tool-stappen; probeer je vraag te verduidelijken."}), 429

    except Exception as e:
        logging.exception("[Clarus error]")
        return jsonify({"error": "Er ging iets mis met Clarus."}), 500


if __name__ == "__main__":
    app.run(debug=True)
