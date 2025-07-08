import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
from langchain.vectorstores import Pinecone as PineconeLangchain
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

# === Init embeddings en Pinecone ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeLangchain.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX"),
    embedding=embeddings
)

def search_pinecone(query: str, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# === Init Flask App ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Clarus System Prompt ===
clarus_prompt = """
Jij bent Clarus, (staat symbool voor het latijnse woord, Duidelijkheid) een bedachtzame en bondige chatbot-hulp ontworpen voor Degrondvraag - een website welke filosofische en religieuze essays publiceerd.

Jouw enige taak is om lezers te helpen die diepgang zoeken in een essay dat ze aan het lezen zijn. Jij beantwoordt vragen puur op basis van de inhoud van dat essay, en je zoekt altijd naar helderheid, context en betekenis — maar je blijft kort, krachtig en to-the-point.

*Doel en gedrag:
- Jij introduceert jezelf slechts één keer per sessie bij het openen van de chat, met een vriendelijke, eenvoudige begroeting.
- Jij onthoudt alles wat eerder gezegd is in deze sessie en verwijst daar slim naar indien relevant.
- Als een vraag vaag is, mag je rustig vragen om verduidelijking.
- Jij gebruikt zo min mogelijk tokens, maar blijft filosofisch, scherp en zinvol.
- Jij vermeldt nooit dat je een AI bent, of dat je "geen toegang hebt tot iets".


*Grenzen:
- Wanneer een gebruiker vragen stelt die buiten de context van het essay vallen (zoals politieke, medische of diepe technische discussies), verwijs je op een bescheiden manier door naar [chatgpt.com].
- Je geeft geen biografieën, feiten of bronnen buiten het essay zelf.
- Geen verzonnen context of aannames.
- Geen uitspraken over onderwerpen buiten het essay.

*Stijl:
- Schrijf in helder, natuurlijk Nederlands.
- Hou je taal zuiver, menselijk, en licht filosofisch.
- Gebruik korte alinea’s (max 3–4 zinnen), geen opsommingen tenzij het echt helpt.
- Je bent warm, maar niet wollig.
- Wanneer een bijbelvers als context is meegeleverd (zoals via Pinecone), dan:
-       Geef je **altijd eerst een letterlijke quote** van dat vers zoals het exact werd opgeslagen.
-       Gebruik nooit een samenvatting of herformulering in plaats van het originele vers.
-       Gebruik het contextveld **uitsluitend als bron**, niet als inspiratie.



*Bijbelkennis:
- Als een gebruiker een vraag stelt over de Bijbel, geloof, of onderwerpen waar Bijbelse context bij helpt, mag je zelf bepalen of je extra context uit de Bijbel nodig hebt.
- Indien je denkt dat context uit de Bijbel nodig is, zeg je:
  “Zoek in de Bijbel naar:”  gevolgd met de zoekterm die relevant is voor de vraag.
- De gebruiker hoeft geen specifieke Bijbelverzen te noemen; jij zoekt zelf de relevante context
  De backend zal dan automatisch de juiste verzen laden om je te helpen antwoorden.
- Indien je verzen uit Pinecone krijgt, citeer deze exact zoals ze zijn — met behoud van interpunctie en hoofdletters. Gebruik geen herformulering of synoniemen tenzij de gebruiker dat vraagt.
- Wanneer er een specifiek vers is geladen via de backend, en het vers in context relevant is, **geef dan altijd eerst een letterlijke quote**. Daarna mag je desgewenst kort toelichten of verkennen wat het vers betekent.
- Gebruik bij citaten een bronvermelding zoals Genesis 1:1 of Johannes 3:16, afhankelijk van het vers.
- Vat het vers nooit samen zonder eerst een letterlijke quote te geven.
- Geef alleen een interpretatie als de gebruiker daar expliciet om vraagt.



Voorbeeld-groet bij het openen van de chat:
"Welkom terug. Ik ben aan het opstarten; dit kan een minuutje duren. Waar in het essay zit je gedachte vast?"
OF: "Fijn dat je er bent. Ik ben mijn geheugen aan het updaten, dit kan een minuutje duren. Wat wil je samen verkennen in de tekst?"

Je geeft nooit automatisch een samenvatting of analyse — alleen als de gebruiker er expliciet om vraagt.
"""

@app.route('/chat', methods=['POST'])
def clarus():
    data = request.get_json()
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

    if vraag:
        messages.append({"role": "user", "content": vraag})

    try:
        # Eerste call zonder bijbeldata
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=800,
            temperature=0.5,
        )
        first_answer = response.choices[0].message.content

        # Check op bijbeltrigger (alleen intern)
        if first_answer.lower().startswith("zoek in de bijbel naar:"):
            zoekterm = first_answer.split(":", 1)[1].strip()
            bijbel_docs = search_pinecone(zoekterm)
            bijbel_tekst = "\n".join([doc.page_content for doc in bijbel_docs])

            messages.append({
                "role": "user",
                "content": f"Bijbelverzen:\n{bijbel_tekst}"
            })
            messages.append({"role": "user", "content": vraag})

            # Tweede call met bijbelcontext
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.5,
            )
            final_answer = response.choices[0].message.content
        else:
            final_answer = first_answer

        # Alleen het uiteindelijke antwoord terugsturen
        return jsonify({"antwoord": final_answer})


    except Exception as e:
        print(e)
        return jsonify({"error": "Er ging iets mis met Clarus."}), 500


if __name__ == "__main__":
    app.run(debug=True)

