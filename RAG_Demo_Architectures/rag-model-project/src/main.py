# -*- coding: utf-8 -*-
"""
Main entry point for the Retrieval-Augmented Generation (RAG) model project.
This script initializes the RAG model, loads the dataset, and orchestrates
the training and evaluation processes.
"""

import pandas as pd
from rag_model import RAGModel
from retriever import Retriever
from generator import Generator

def main():
    # Load the dataset
    dataset_path = 'data/dataset.csv'
    data = pd.read_csv(dataset_path)

    # Initialize the retriever and generator
    retriever = Retriever(data)
    generator = Generator()

    # Initialize the RAG model
    rag_model = RAGModel(retriever, generator)

    # Train the model
    rag_model.train()

    # Evaluate the model
    results = rag_model.evaluate()
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()