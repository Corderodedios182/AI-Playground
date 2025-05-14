def preprocess_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    # Perform any necessary preprocessing steps
    return df

def calculate_metrics(true_labels, predicted_labels):
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, f1

def log_message(message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(message)