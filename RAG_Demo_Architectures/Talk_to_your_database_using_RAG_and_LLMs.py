# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:26:17 2025

@author: corde
"""
import ollama

def generate_query_with_ollama(question):
    system_prompt = """
    You are a data analysis assistant. You will receive a question about a Bitcoin dataset.
    Generate a valid Python Pandas query to analyze the data. The dataset has the following columns:
    - Date (string, format: MM/DD/YYYY)
    - Price (float, closing price of Bitcoin)
    - Open (float, opening price)
    - High (float, highest price of the day)
    - Low (float, lowest price of the day)
    - Vol. (float, trading volume)
    - Change % (float, percentage change)
    
    Example Questions and Queries:
    - "What was the highest Bitcoin price in 2024?" → df.loc[df["High"].idxmax()]
    - "What is the average Bitcoin price?" → df["Price"].mean()
    - "What was the total trading volume?" → df["Vol."].sum()
    
    Generate only the Pandas query, without explanations.
    """
    
    response = ollama.chat(
        model="mistral",  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    
    query = response["message"]["content"].strip()
    return query

question_test = "What was the highest Bitcoin price recorded in 2024?"

generated_query = generate_query_with_ollama(question_test)

try:
    result_ollama = eval(generated_query)
except Exception as e:
    result_ollama = f"Error executing query: {e}"

result_ollama
