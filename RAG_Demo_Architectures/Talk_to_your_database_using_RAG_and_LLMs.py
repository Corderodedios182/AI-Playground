# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:26:17 2025

@author: corde
"""
import os
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Configura la API

OPENAI_API_KEY = os.getenv("OpenIA")

llm = ChatOpenAI(temperatura= 0 , openai_api_key=OPENAI_API_KEY)

host = 'localhost'
port = '3306'
username = 'root'
password = 'password'
database_schema = 'agency_db'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"

#include_tables : solo hacemos referencia a la tabla job_details y realizamos consultas únicamente sobre esta tabla
#sample_rows_in_table_info : al interactuar con la base de datos, langchain recibirá las 2 primeras filas de esta tabla como muestra para un contexto adicional y una mejor recuperación
db = SQLDatabase.from_uri(mysql_uri, include_tables=['job_details'],sample_rows_in_table_info=2)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def retrieve_from_db(query: str) -> str:
    db_context = db_chain(query)
    db_context = db_context['result'].strip()
    return db_context

#
def generate(query: str) -> str:
    db_context = retrieve_from_db(query)
    
    system_message = """You are a professional representative of an employment agency.
        You have to answer user's queries and provide relevant information to help in their job search. 
        Example:
        
        Input:
        Where are the most number of jobs for an English Teacher in Canada?
        
        Context:
        The most number of jobs for an English Teacher in Canada is in the following cities:
        1. Ontario
        2. British Columbia
        
        Output:
        The most number of jobs for an English Teacher in Canada is in Toronto and British Columbia
        """
    
    human_qry_template = HumanMessagePromptTemplate.from_template(
        """Input:
        {human_input}
        
        Context:
        {db_context}
        
        Output:
        """
    )
    messages = [
      SystemMessage(content=system_message),
      human_qry_template.format(human_input=query, db_context=db_context)
    ]
    response = llm(messages).content
    return response

generate("")


