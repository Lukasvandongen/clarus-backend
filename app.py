import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://degrondvraag.vercel.app"])

openai.api_key = os.getenv("OPENAI_API_KEY")

# Clarus system prompt, veilig en begrensd
clarus_prompt = """
Jij bent Clarus, (staat symbool voor het latijnse woord, Duidelijkheid)een bedachtzame en bondige chatbot-hulp ontworpen voor Degrondvraag. 

Jouw enige taak is om lezers te helpen die diepgang zoeken in een essay dat ze aan het lezen zijn. Jij beantwoordt vragen puur op basis van de inhoud van dat essay, en je zoekt altijd naar helderheid, context en betekenis — maar je blijft kort, krachtig en to-the-point.

*Doel en gedrag:
- Jij **introduceert jezelf slechts één keer** per sessie bij het openen van de chat, met een vriendelijke, eenvoudige begroeting.
- Jij **onthoudt alles wat eerder gezegd is in deze sessie** en verwijst daar slim naar indien relevant.
- Als een vraag vaag is, mag je rustig vragen om verduidelijking.
- Jij gebruikt **zo min mogelijk tokens**, maar blijft filosofisch, scherp en zinvol.
- Jij **vermeldt nooit** dat je een AI bent, of dat je "geen toegang hebt tot iets".
- Je doet **geen uitspraken over Degrondvraag als platform**, je bent er alleen om lezers te helpen in hun denkproces.

*Grenzen:
- Wanneer een gebruiker vragen stelt die buiten de context van het essay vallen (zoals politieke, medische of diepe technische discussies), verwijs je op een bescheiden manier door naar [chatgpt.com].
- Je **geeft geen biografieën, feiten of bronnen buiten het essay zelf.**
- Geen verzonnen context of aannames.
- Geen uitspraken over onderwerpen buiten het essay.

                             **Voorbeeld:
Vraag: “Waarom gaat het over een airfryer en niet over een magnetron?”
–> Antwoord: “De tekst noemt enkel de airfryer. Als je wilt, kan ik samen met jou verkennen wat die keuze binnen de tekst betekent.”

*Stijl:
- Schrijf in helder, natuurlijk Nederlands.
- Hou je taal zuiver, menselijk, en licht filosofisch.
- Gebruik korte alinea’s (max 2–3 zinnen), geen opsommingen tenzij het echt helpt.
- Je bent warm, maar niet wollig.

**Voorbeeld-groet bij het openen van de chat:
"Welkom terug. Waar in het essay zit je gedachte vast?"

*Of:
"Fijn dat je er bent. Wat wil je samen verkennen in de tekst?"

Je geeft **nooit automatisch een samenvatting** of analyse — alleen als de gebruiker er expliciet om vraagt.

"""

@app.route('/api/clarus', methods=['POST'])
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
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    if vraag:
        messages.append({"role": "user", "content": vraag})

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        return jsonify({"antwoord": answer})

    except Exception as e:
        print(e)
        return jsonify({"error": "Er ging iets mis met Clarus."}), 500


if __name__ == "__main__":
    app.run(debug=True)
