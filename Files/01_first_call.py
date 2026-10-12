from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
    {"role": "system", "content": "Tu es un assistant juridique français. Réponds en 3 phrases maximum."},
    {"role": "user", "content": "Quel article du Code civil définit le contrat ?"},
    ],
    temperature=0,
)
# TODO: print only the answer text (explore the response object to find it)
print(response.choices[0].message.content)
