import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# Sample Data
data = {
'user_id': [1, 1, 2, 2, 3, 3],
'movie_id': [1, 2, 1, 3, 2, 4],
'rating': [5, 3, 4, 2, 5, 4]
}
df = pd.DataFrame(data)
# Movie Metadata
movies = {
'movie_id': [1, 2, 3, 4],
'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight']
}
movies_df = pd.DataFrame(movies)
# Create User-Item Matrix
user_item_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
print("User-Item Matrix:")
print(user_item_matrix)
# Compute User Similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index,columns=user_item_matrix.index)
print("\nUser Similarity Matrix:")
print(user_similarity_df)
# Recommendation Function
def get_recommendations(user_id, user_item_matrix, user_similarity_df):
  # Get similar users
  similar_users = user_similarity_df[user_id].sort_values(ascending=False).index
  # Get user ratings
  user_ratings = user_item_matrix.loc[user_id]
  # Calculate weighted average of ratings
  weighted_ratings = np.zeros(user_item_matrix.shape[1])
  for i, movie_id in enumerate(user_item_matrix.columns):
    # Get ratings of the movie from similar users
    similar_ratings = user_item_matrix.loc[similar_users].iloc[:, i]
    # Calculate weighted average
    if similar_ratings.sum() != 0:
      weighted_ratings[i] = similar_ratings.dot(user_similarity_df[user_id]) /user_similarity_df[user_id].sum()
  # Create DataFrame for recommendations
  recommendations = pd.DataFrame({
    'movie_id': user_item_matrix.columns,
    'predicted_rating': weighted_ratings
  }).sort_values(by='predicted_rating', ascending=False)
  # Merge with movie titles
  recommendations = recommendations.merge(movies_df, on='movie_id’)
  return recommendations
# Plot Recommendations
def plot_recommendations(recommendations):
  plt.figure(figsize=(10, 6))
  plt.barh(recommendations['title'], recommendations['predicted_rating'], color='skyblue')
  plt.xlabel('Predicted Rating')
  plt.ylabel('Movie Title')
  plt.title('Top Movie Recommendations')
  plt.gca().invert_yaxis() # Highest ratings at the top
  plt.show()
# Example Recommendation for User 1
recommendations = get_recommendations(user_id=1, user_item_matrix=user_item_matrix,
user_similarity_df=user_similarity_df)
print("\nRecommendations for User 1:")
print(recommendations)
# Plot the recommendations
plot_recommendations(recommendations)
