import requests
import os

SNOWFLAKE_API_URL = "https://abc32220.us-east-1.snowflakecomputing.com/api/v2/databases/PDF_KNOWLEDGE_BASE_V3/schemas/DOC_SEARCH/agents/PDF_DECISION_DB_AGENT:run"

SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN_BERNARDO")

if not SNOWFLAKE_TOKEN:
    raise ValueError("No se encontró la variable de entorno SNOWFLAKE_TOKEN")

def extract_answer(data: dict) -> str:
    answer_parts = []

    for item in data.get("content", []):
        if item.get("type") == "text" and item.get("text"):
            answer_parts.append(item["text"])

    return "\n".join(answer_parts).strip()


question = "Escribe tu pregunta: Dame un resumen sobre los datos mtcars"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ],
    "stream": False
}

headers = {
    "Authorization": f"Bearer {SNOWFLAKE_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(
    SNOWFLAKE_API_URL,
    json=payload,
    headers=headers,
    timeout=60
)

data = response.json()
answer = extract_answer(data)

print("\nRespuesta:\n")
print(answer)

