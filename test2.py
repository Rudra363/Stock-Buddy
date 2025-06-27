import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import joblib

def load_embeddings(csv_path):
    """
    Loads embeddings from a CSV file and parses them into a numpy array.
    Assumes the CSV has columns 'title' and 'embedding'.
    """
    df = pd.read_csv(csv_path)
    embeddings = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    return df['embedding'].tolist(), np.vstack(embeddings)

def initialize_model(model_name='all-MiniLM-L6-v2'):
    """
    Initializes and returns the sentence transformer model.
    """
    return SentenceTransformer(model_name)

def train_regressor(embeddings, impact_percentages):
    """
    Trains a RandomForestRegressor on the embeddings to predict impact percentages.
    """
    regressor = RandomForestRegressor()
    regressor.fit(embeddings, impact_percentages)
    return regressor

def find_most_similar_title(new_title, titles, embeddings, model):
    """
    Finds the most similar title using cosine similarity.
    """
    new_embedding = model.encode([new_title])
    similarities = cosine_similarity(new_embedding, embeddings)
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0][most_similar_idx]
    return most_similar_idx, similarity_score

def run3(stock, threshold=0.55):
    csv_path = 'embeddedNews.csv'
    model = initialize_model()
    news_items = stock.getNews()
    added_percent = 0.0
    count = 0

    # Load dataset
    titles, embeddings = load_embeddings(csv_path)
    impact_data = None
    try:
        df = pd.read_csv(csv_path)
        impact_data = df['percent_change'].tolist()
    except Exception:
        pass  # No impact data available

    model_path = 'news_impact_regressor.pkl'
    regressor = None
    if impact_data:
        # Load existing model if available
        if os.path.exists(model_path):
            #print("Exists")
            regressor = joblib.load(model_path)
        else:
            # Train and save
           # print("Training...")
            regressor = train_regressor(embeddings, impact_data)
            joblib.dump(regressor, model_path)

    # Find most similar title
    for item in news_items:
        title = item.get("content", {}).get("title")
        if not title:
            continue

        idx, similarity = find_most_similar_title(title, titles, embeddings, model)

       # print(f"Input Title: '{title}'")
       # print(f"Most similar title in dataset: '{titles[idx]}'")
        #print(f"Similarity score: {similarity:.2f}")

    # if similarity >= similarity_threshold and regressor:
    #     new_embedding = model.encode([new_title])
    #     predicted_impact = regressor.predict(new_embedding)[0]
    #     print(f"Predicted impact on stock: {predicted_impact:.2f}%")
    # else:
    #     print("No sufficiently similar title found in dataset or impact data missing.")

        if similarity >= threshold:
            new_embedding = model.encode([title])
            predicted_impact = regressor.predict(new_embedding)[0]
           # print(f"Predicted impact on stock: {predicted_impact:.2f}%")
            added_percent += predicted_impact
            count += 1
            return True
        else:
            pass
            #print("No sufficiently similar title found in dataset.")

    if count == 0:
        return False, 0.0

    final_value = added_percent / count
    print(f"DEBUG run3() output: {final_value}")
    print(bool(final_value > 0))
    return bool(final_value > 0)
    #, float(round(final_value, 2))
