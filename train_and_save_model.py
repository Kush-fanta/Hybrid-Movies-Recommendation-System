import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import svds
import joblib
import pickle

print("--- Starting Model Training and Saving ---")
start_time = time.time()

# --- Step 1: Load and Clean Your Data ---
print("Step 1: Loading and cleaning data...")
try:
    df_movies = pd.read_csv('movies.csv')
    df_ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    print("Error: 'movies.csv' or 'ratings.csv' not found. Exiting.")
    exit()

valid_movie_ids = set(df_movies['movieId'])
df_ratings_clean = df_ratings[df_ratings['movieId'].isin(valid_movie_ids)]
print(f"Cleaned ratings count: {len(df_ratings_clean)}")

# --- Create and Save the "Seen Movies" Map ---
print("Creating 'seen movies' map...")
# This creates a dict like {userId: [movieId1, movieId2, ...]}
seen_movies_map = df_ratings_clean.groupby('userId')['movieId'].apply(list).to_dict()
with open('seen_movies_map.pkl', 'wb') as f:
    pickle.dump(seen_movies_map, f)
print("Saved 'seen_movies_map.pkl'")


# --- Step 2: Build and Save Content-Based Model ---
print("\nStep 2: Building Content-Based model...")
df_movies['genres'] = df_movies['genres'].str.replace('|', ' ', regex=False).fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['genres'])
indices_map = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

print("Saving Content-Based artifacts...")
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
save_npz('tfidf_matrix.npz', tfidf_matrix)
with open('indices_map.pkl', 'wb') as f:
    pickle.dump(indices_map, f)
print("Saved 'tfidf_vectorizer.joblib', 'tfidf_matrix.npz', and 'indices_map.pkl'")


# --- Step 3: Build and Save Popularity-Based Model ---
print("\nStep 3: Building Popularity-Based model...")
movie_stats = df_ratings_clean.groupby('movieId').agg(mean_rating=('rating', 'mean'), rating_count=('rating', 'count')).reset_index()
df_popular = df_movies.merge(movie_stats, on='movieId', how='left')
C = df_popular['mean_rating'].mean()
m = df_popular['rating_count'].quantile(0.90)

def weighted_rating(x, m=m, C=C):
    v = x['rating_count']
    R = x['mean_rating']
    # Avoid division by zero if v+m is 0 (though m is a quantile, so unlikely)
    if (v + m) == 0:
        return C 
    return (v / (v + m) * R) + (m / (v + m) * C)

df_popular['score'] = df_popular.apply(weighted_rating, axis=1)
df_popular = df_popular.sort_values('score', ascending=False)

print("Saving Popularity artifact...")
df_popular.to_pickle('popular_movies_df.pkl')
print("Saved 'popular_movies_df.pkl'")


# --- Step 4: Build and Save Collaborative Filtering Model ---
print("\nStep 4: Building Collaborative Filtering model...")
user_ids = df_ratings_clean['userId'].unique()
user_id_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
n_users = len(user_ids)

movie_ids = df_movies['movieId'].unique()
movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
n_movies = len(movie_ids)

# Remap dataframes to internal indices
df_ratings_clean['user_idx'] = df_ratings_clean['userId'].map(user_id_to_idx)
df_ratings_clean['movie_idx'] = df_ratings_clean['movieId'].map(movie_id_to_idx)

# Drop rows with NaN indices (if any movie was in ratings but not movies, though we cleaned this)
df_ratings_clean = df_ratings_clean.dropna(subset=['user_idx', 'movie_idx'])

# Convert indices to int
df_ratings_clean['user_idx'] = df_ratings_clean['user_idx'].astype(int)
df_ratings_clean['movie_idx'] = df_ratings_clean['movie_idx'].astype(int)


print("Creating the [User x Movie] utility matrix...")
utility_matrix = csr_matrix((df_ratings_clean['rating'],
                           (df_ratings_clean['user_idx'], df_ratings_clean['movie_idx'])),
                           shape=(n_users, n_movies))
print("Running SVD...")
U, S, Vt = svds(utility_matrix, k=50)
S_diag = np.diag(S)
predicted_ratings_matrix = U @ S_diag @ Vt

print("Saving Collaborative Filtering artifacts...")
np.save('predicted_ratings_matrix.npy', predicted_ratings_matrix)
with open('user_id_to_idx.pkl', 'wb') as f:
    pickle.dump(user_id_to_idx, f)
print("Saved 'predicted_ratings_matrix.npy' and 'user_id_to_idx.pkl'")

print(f"\n--- All models trained and saved successfully in {(time.time() - start_time):.2f} seconds ---")
