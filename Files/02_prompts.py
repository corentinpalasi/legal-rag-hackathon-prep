from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def ask(question, system_prompt):
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        ]

    response = client.chat.completions.create( model="llama3.2", messages= messages, temperature=0)

    return response.choices[0].message.content

    # TODO: build the messages list from system_prompt and question
    # TODO: call client.chat.completions.create with temperature=0
    # TODO: return the answer text (the path you found)


#print(ask("Quel article du Code pénal punit le vol ?", "Tu es un assistant juridique français."))
#print(ask("Quel article du Code pénal punit le vol ?", "Réponds en une phrase. N'invente jamais un numéro d'article. Si tu n'es pas sûr, réponds 'Je ne sais pas'."))

#Generic prompt: “article 226”, wrong every time (it’s 311-3). Confidently wrong.
#Strict prompt: “Je ne sais pas”. Honest, but useless to a user.


source = """Article 311-1 du Code pénal : Le vol est la soustraction frauduleuse de la chose d'autrui.
Article 311-3 du Code pénal : Le vol est puni de trois ans d'emprisonnement et de 45 000 euros d'amende."""

strict_with_source = "Réponds uniquement à partir des sources ci-dessous et cite l'article. Si la réponse n'y est pas, réponds 'Je ne sais pas'.\n\nSOURCES :\n" + source

print(ask("Quel article du Code pénal punit le vol, et quelle est la peine ?", strict_with_source))
print(ask("Quelle est la peine pour une escroquerie ?", strict_with_source))
