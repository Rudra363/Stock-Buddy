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

def newTitle(stock):
    # datasetEmbedding, impact = readEmbedded()
    news_items = stock.getNews()
    trueValues = 0
    falseValues = 0
    addedPercent = 0.0

    for item in news_items:
        try:
            newTitle = item["content"]["title"]
        except KeyError:
            continue

        new_embedding = model.encode([newTitle])

        similarities = cosine_similarity(new_embedding, dataset_embeddings)
        most_similar_idx = similarities.argmax()
        most_similar_impact = impact_values[most_similar_idx]

        similarity_score = similarities[0][most_similar_idx]

        threshold = 0.55

        print(f"Input Title: '{newTitle}'")
        print(f"Similarity score: {similarity_score:.2f}")

        if similarity_score >= threshold:
            predicted_impact = regressor.predict(new_embedding)[0]
            print(f"Predicted impact on stock: {predicted_impact:.2f}%")
            addedPercent += most_similar_impact

        else:
            print("No sufficiently similar title found in dataset.")

        # if most_similar_impact < 0:
        #     falseValues += 1
        # else:
        #     trueValues += 1

    if addedPercent > 0:
        return True, addedPercent
    else:
        return False, addedPercent


# stock = Stocks("AAPL", 10)
# readEmbedded()
# print(newTitle(stock))