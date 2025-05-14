class Retriever:
    def __init__(self, dataset_path):
        import pandas as pd
        self.dataset = pd.read_csv(dataset_path)
        self.indexed_data = self.index_data()

    def index_data(self):
        # Create an index for fast retrieval
        indexed_data = {}
        for idx, row in self.dataset.iterrows():
            # Assuming the dataset has a 'query' column for indexing
            query = row['query']
            indexed_data[query] = row
        return indexed_data

    def retrieve(self, query):
        # Retrieve relevant data based on the input query
        return self.indexed_data.get(query, None)