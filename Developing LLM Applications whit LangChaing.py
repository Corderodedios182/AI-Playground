
#Langchaing have 3 principal components : 
    #LLM seleccionado código abierto (hugging faces) o código cerrado (Opena IA, AWS Bedrock)

#Hugging Face models in LangChain!
import os
from langchain_community.llms import HuggingFaceHub

# Set your Hugging Face API token 
huggingfacehub_api_token = os.getenv("huggingfacehub_api_token_Cordero")

# Define the LLM
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = '¿Como estar solo?'
output = llm.predict(question)

print(output)
