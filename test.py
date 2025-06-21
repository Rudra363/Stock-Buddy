
# --------------------------
# TODO: The code below works and should bne used later

# from sentence_transformers import SentenceTransformer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# # Initialize the sentence embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# # Dataset of titles and their impacts (% change in stock price)
# dataset_titles = [
#     "Apple hits new high",
#     "Market crashes unexpectedly",
#     "Tech stocks rally",
#     "Inflation worries grow",
#     "New product launch excites investors"
# ]
# impact_percentages = [3.2, -2.5, 4.1, -1.8, 2.9]  # example impact values
#
# # Encode dataset titles
# dataset_embeddings = model.encode(dataset_titles)
#
# # Train a regression model to predict impact based on embeddings
# regressor = RandomForestRegressor()
# regressor.fit(dataset_embeddings, impact_percentages)
#
# # New input title
# # new_title = "Apple reaches record sales"
# new_title = "Tech market booms"
#
# # Encode the new title
# new_embedding = model.encode([new_title])
#
# # Calculate cosine similarity to find the most similar title
# similarities = cosine_similarity(new_embedding, dataset_embeddings)
# most_similar_idx = similarities.argmax()
#
# # Fetch the most similar title and its similarity score
# most_similar_title = dataset_titles[most_similar_idx]
# similarity_score = similarities[0][most_similar_idx]
#
# # Define a similarity threshold (e.g., 0.75 for strong similarity)
# threshold = 0.75
#
# print(f"Input Title: '{new_title}'")
# print(f"Most similar title in dataset: '{most_similar_title}'")
# print(f"Similarity score: {similarity_score:.2f}")
#
# if similarity_score >= threshold:
#     # Predict impact for the new title
#     predicted_impact = regressor.predict(new_embedding)[0]
#     print(f"Predicted impact on stock: {predicted_impact:.2f}%")
# else:
#     print("No sufficiently similar title found in dataset.")