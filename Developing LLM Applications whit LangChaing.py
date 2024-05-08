import os
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Set your API token 
huggingfacehub_api_token = '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'#os.getenv('huggingfacehub_api_token_Cordero')
openai_api_key = os.getenv("OpenaAI_api_token_Cordero")

# Define the LLM
llm_Hfh = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)
llm_OpenAI = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)

# Predict the words following the text in question
question = '¿Quienes fueron los Aztecas?'
output_Hfh = llm_Hfh.predict(question)
output_OpenAI = llm_OpenAI.predict(question)

print(output_Hfh)
print(output_OpenAI)

#-- Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=['question'])

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Create a chain to integrate the prompt template and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm_Hfh)
llm_chat_OpenAI = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# Insert a question into the template and call the model
question = "¿Quienes fueron los Aztecas?"
print(llm_chain.run(question))

full_prompt = prompt_template.format_messages(question='How can I retain learning?')
llm_chat_OpenAI(full_prompt)
