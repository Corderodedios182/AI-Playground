"""
Con el SDK de Azure AI Foundry, puede conectarse a un proyecto y recuperar conexiones; que puede usar para consumir los servicios conectados.

Por ejemplo, el objeto AIProjectClient de Python tiene una propiedad connections , que puede usar para acceder a las conexiones de recursos en el proyecto.

Los métodos de las conexiones objeto incluyen:

 ```connections.list() ```  : devuelve una colección de objetos de conexión, cada uno que representa una conexión en el proyecto. 

Puede filtrar los resultados especificando un parámetro de connection_type opcional con una enumeración válida, como ConnectionType.AZURE_OPEN_AI.

 ```connections.get(connection_name, include_credentials) ``` : devuelve un objeto de conexión para la conexión con el nombre especificado. 

Si el parámetro include_credentials es True (el valor predeterminado), se devuelven las credenciales necesarias para conectarse a la conexión; por ejemplo, en forma de clave de API para un recurso de servicios de Azure AI.

Los objetos de conexión devueltos por estos métodos incluyen propiedades específicas de la conexión, incluidas las credenciales, que puede usar para conectarse al recurso asociado.

En el ejemplo de código siguiente se enumeran todas las conexiones de recursos que se han agregado a un proyecto:

Se usa el método ``` get_openai_client()```  para obtener un cliente de OpenAI con el que chatear con un modelo que se ha implementado en el recurso de Azure AI Foundry del proyecto.

```
python -m venv labenv
./labenv/bin/Activate.ps1
pip install -r requirements.txt azure-identity azure-ai-projects openai
``` 

"""

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

#pip install azure-ai-projects
#pip install azure-identity
#pip install openai

try:
    
    # connect to the project
    project_endpoint = "https://......"
    project_client = AIProjectClient(            
            credential=DefaultAzureCredential(),
            endpoint=project_endpoint,
        )
    
    ## List all connections in the project
    connections = project_client.connections
    print("List all connections:")
    for connection in connections.list():
        print(f"{connection.name} ({connection.type})")
    
    # Get a chat client
    chat_client = project_client.get_openai_client(api_version="2024-10-21")
    
    # Get a chat completion based on a user-provided prompt
    user_prompt = input("Enter a question:")
    
    response = chat_client.chat.completions.create(
        model=your_model_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response.choices[0].message.content)

except Exception as ex:
    print(ex)
    