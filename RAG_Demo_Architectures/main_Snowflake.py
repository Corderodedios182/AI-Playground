import requests
import os

SNOWFLAKE_API_URL = "https://abc32220.us-east-1.snowflakecomputing.com/api/v2/databases/PDF_KNOWLEDGE_BASE_V3/schemas/DOC_SEARCH/agents/PDF_DECISION_DB_AGENT:run"

SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN_BERNARDO")

if not SNOWFLAKE_TOKEN:
    raise ValueError("No se encontró la variable de entorno SNOWFLAKE_TOKEN_BERNARDO")


def extract_answer(data: dict) -> str:
    answer_parts = []

    for item in data.get("content", []):
        if item.get("type") == "text" and item.get("text"):
            answer_parts.append(item["text"])

    return "\n".join(answer_parts).strip()


def call_agent(history: list[dict]) -> dict:
    payload = {
        "messages": history,
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

    response.raise_for_status()
    return response.json()


def main():
    print("=== Chat con Snowflake Agent ===")
    print("Escribe 'salir' para terminar.\n")

    history = []

    while True:
        question = input("Tú: ").strip()

        if not question:
            print("Por favor escribe una pregunta.\n")
            continue

        if question.lower() in ["salir", "exit", "quit"]:
            print("Chat finalizado.")
            break

        # Agregar mensaje del usuario al historial
        history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                }
            ]
        })

        try:
            data = call_agent(history)
            answer = extract_answer(data)

            if not answer:
                answer = "No se encontró respuesta en el contenido devuelto por el agente."

            print(f"\nAgente:\n{answer}\n")

            # Guardar respuesta del agente en historial
            history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer
                    }
                ]
            })

        except requests.exceptions.HTTPError as e:
            print(f"\nError HTTP: {e.response.status_code} - {e.response.text}\n")

        except requests.exceptions.RequestException as e:
            print(f"\nError de conexión: {str(e)}\n")

        except Exception as e:
            print(f"\nError inesperado: {str(e)}\n")


if __name__ == "__main__":
    main()
