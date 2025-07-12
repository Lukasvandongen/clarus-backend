import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import qdrant_client
from qdrant_client.http import models as qmodels
import uuid
from langchain.embeddings import OpenAIEmbeddings
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

qdrant = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "bijbel"


def search_qdrant(query: str, k=5):
    embedding_response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    vector = embedding_response.data[0].embedding

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=k,
        with_payload=True
    )

    class FakeDoc:
        def __init__(self, content, boek, hoofdstuk, vers):
            self.page_content = content
            self.boek = boek
            self.hoofdstuk = hoofdstuk
            self.vers = vers

    return [
        FakeDoc(
            content=hit.payload.get("tekst", ""),
            boek=hit.payload.get("boek", ""),
            hoofdstuk=hit.payload.get("hoofdstuk", ""),
            vers=hit.payload.get("vers", "")
        )
        for hit in hits
        if "tekst" in hit.payload
    ]


# === Init Flask App ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://www.degrondvraag.com", "https://degrondvraag.com"]}})
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Clarus System Prompt ===
clarus_prompt = """
Jij bent Clarus(latijn voor 'duidelijkheid', een bedachtzame en bondige chatbot-hulp ontworpen voor Degrondvraag — een website die filosofische en religieuze essays publiceert door een anonieme jonge man.

Je helpt lezers die verdieping zoeken in een specifiek essay. Je antwoorden baseer je **uitsluitend** op de inhoud van het essay en op meegeleverde Bijbelverzen. Gebruik nooit eigen kennis of externe bronnen. Je antwoorden zijn kort, krachtig en helder van toon.

— Gedrag —
- Je begroet gebruikers alleen bij het begin van de sessie.
- Je verwijst slim terug naar eerdere vragen binnen dezelfde sessie.
- Je stelt gerust een vervolgvraag als de input vaag is.
- Je gebruikt weinig tokens, maar blijft filosofisch en zuiver.
- Nooit vermelden dat je een AI bent.

— Grenzen —
- Alleen reageren op vragen die binnen het essay of de meegeleverde Bijbelcontext vallen.
- Geen externe feiten, bronnen of verzinsels.
- Geen uitspraken over onderwerpen buiten de geleverde context.
- Je mag nooit zelf een Bijbelvers verzinnen, samenvatten of parafraseren.
- Als een vraag om Bijbelcontext vraagt, zeg dan enkel: `Bijbel nodig: [zoekterm of vers]`
- Pas wanneer je daarna verzen ontvangt, mag je die analyseren.

— Stijl —
- Natuurlijk en helder Nederlands.
- Licht filosofisch, maar niet wollig.
- Je gaat met de taal en het niveau van de gebruiker mee.
- Geen opsommingen, tenzij functioneel.
- Bij context uit de Bijbel geldt:
  - Citeer eerst letterlijk het vers (zoals aangeleverd).
  - Geef daarna pas uitleg, enkel als dat gevraagd wordt.
  - Nooit herformuleren, altijd interpunctie en hoofdletters behouden.
  - Voeg bron toe (bijv. Genesis 1:1).

— Bijbelinteractie —
- Je bepaalt zelf of er Bijbelcontext nodig is.
- Als je dat denkt, geef je: `Zoek in de Bijbel naar: [zoekterm]`
- Je doet dit alleen als je onvoldoende aan de context hebt.
- Na ontvangen verzen mag je antwoorden.

Voorbeeldgroet bij opstarten:
- "Welkom terug. Ik ben aan het opstarten; dit kan een minuutje duren. Waar in het essay zit je gedachte vast?"
- "Fijn dat je er bent. Ik update mijn geheugen; dit kan een minuutje duren. Wat wil je samen verkennen in de tekst?"

Geef nooit automatisch een analyse of samenvatting zonder dat er expliciet om gevraagd is.
"""


@app.route('/chat', methods=['POST'])
def clarus():
    data = request.get_json()
    if not data:
        return jsonify({"antwoord": "Geen geldige data ontvangen"}), 400
    essay = data.get('essay', '')
    vraag = data.get('vraag', '')
    history = data.get('history', [])

    messages = [
        {"role": "system", "content": clarus_prompt},
        {"role": "user", "content": f"Essay: {essay}"}
    ]

    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            messages.append(msg)

    if not data:
        return jsonify({"antwoord": "Geen geldige data ontvangen"}), 400

    try:
        if vraag:
            messages.append({"role": "user", "content": vraag})
            logging.info(f"[Vraag ontvangen] {vraag}")

            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.5,
            )
            first_answer = response.choices[0].message.content
            logging.info(f"[Clarus eerste antwoord] {first_answer}")

            bijbel_trigger = None
            if "bijbel nodig:" in first_answer.lower():
                bijbel_trigger = first_answer.split(":", 1)[1].strip()
                logging.info(f"[Bijbel context trigger herkend] {bijbel_trigger}")

                bijbel_docs = search_qdrant(bijbel_trigger)
                logging.info(f"[Bijbelverzen geladen] {len(bijbel_docs)} resultaten")

                for doc in bijbel_docs:
                    logging.info(f"[Vers] {doc.boek} {doc.hoofdstuk}:{doc.vers} – {doc.page_content[:60]}...")

                bijbel_tekst = "\n".join(
                    [f'{doc.boek} {doc.hoofdstuk}:{doc.vers} — {doc.page_content}' for doc in bijbel_docs]
                )

                messages.append({
                    "role": "user",
                    "content": f"Bijbelverzen:\n{bijbel_tekst}"
                })
                messages.append({"role": "user", "content": vraag})

                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.5,
                )
                final_answer = response.choices[0].message.content
            else:
                final_answer = first_answer

            logging.info(f"[Clarus eindantwoord] {final_answer}")
            return jsonify({"antwoord": final_answer})

        else:
            return jsonify({"antwoord": "Geen vraag ontvangen"}), 400

    except Exception as e:
        logging.error(f"[Clarus error] {str(e)}")
        return jsonify({"error": "Er ging iets mis met Clarus."}), 500



if __name__ == "__main__":
    app.run(debug=True)

