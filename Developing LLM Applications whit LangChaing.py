from langchain_community.llms import HuggingFaceHub

# Set your Hugging Face API token 
huggingfacehub_api_token = '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'

# Define the LLM
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.predict(question)

print(output)
