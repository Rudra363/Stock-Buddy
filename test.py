from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import ast
import csv
from StocksClass import Stocks

# Initialize the sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
regressor = RandomForestRegressor()
dataset_embeddings = []
impact_values = []

def readEmbedded():
    global dataset_embeddings, impact_values, regressor
    embedding = []
    impact = []
    with open("embeddedNews.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            title = ast.literal_eval(row[1])
            impactPercentage = float(row[2])
            embedding.append(title)
            impact.append(impactPercentage)
    dataset_embeddings = embedding
    impact_values = impact
    regressor.fit(embedding, impact)

    return embedding, impact

def newTitle(stock, threshold=0.55):
    if not dataset_embeddings or not impact_values:
        readEmbedded()

    news_items = stock.getNews()
    added_percent = 0.0
    count = 0

    for item in news_items:
        title = item.get("content", {}).get("title")
        if not title:
            continue

        new_embedding = model.encode([title])
        similarities = cosine_similarity(new_embedding, dataset_embeddings)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0][most_similar_idx]

        print(f"Input Title: '{title}'")
        print(f"Similarity score: {similarity_score:.2f}")

        if similarity_score >= threshold:
            predicted_impact = regressor.predict(new_embedding)[0]
            print(f"Predicted impact on stock: {predicted_impact:.2f}%")
            added_percent += predicted_impact
            count += 1
        else:
            print("No sufficiently similar title found in dataset.")

    if count == 0:
        return False, 0.0

    final_value = added_percent / count
    return bool(final_value > 0), float(round(final_value, 2))


stock = Stocks("AAPL", 10)
# readEmbedded()
print(newTitle(stock))