import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel
import uuid # To create a unique session ID
import random

@st.cache_resource
def load_all_models():
    """
    Loads all pre-trained models and artifacts from disk.
    This now includes the 'tfidf_vectorizer'.
    """
    print("Loading models... This will run only once.")
    
    # Loading Movie Data
    df_movies = pd.read_csv('data/movies.csv')
    n_movies = len(df_movies)
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(df_movies['movieId'])}
    
    # Loading Content-Based Filtering Model
    tfidf_matrix = load_npz('tfidf_matrix.npz')
    with open('indices_map.pkl', 'rb') as f:
        indices_map = pickle.load(f)
        

    # Loading the TF-IDF Vectorizer
    try:
        tfidf_vectorizer = joblib.load('content_based_tfidf_vectorizer.joblib')
        all_genres = tfidf_vectorizer.get_feature_names_out()
    except FileNotFoundError:
        print("ERROR: 'tfidf_vectorizer.joblib' not found.")
        return None

    # Loading Popularity Model
    df_popular = pd.read_pickle('popular_movies_df.pkl')
    popular_movies_list = df_popular['title'].values

    # Loading Collaborative Filtering Model 
    predicted_ratings_matrix = np.load('predicted_ratings_matrix.npy')
    with open('user_id_to_idx.pkl', 'rb') as f:
        user_id_to_idx = pickle.load(f)
        
    # Loading Seen Movies Map
    with open('seen_movies_map.pkl', 'rb') as f:
        seen_movies_map = pickle.load(f)

    general_cf_scores = np.mean(predicted_ratings_matrix, axis=0)
    general_cf_scores_normalized = (general_cf_scores - 0.5) / 4.5

    print("...Models loaded successfully.")


    return {
        'df_movies': df_movies, 'n_movies': n_movies, 'movie_id_to_idx': movie_id_to_idx,
        'tfidf_matrix': tfidf_matrix, 'indices_map': indices_map, 
        'popular_movies_list': popular_movies_list, 'user_id_to_idx': user_id_to_idx,
        'predicted_ratings_matrix': predicted_ratings_matrix, 'seen_movies_map': seen_movies_map,
        'general_cf_scores': general_cf_scores_normalized,
        'tfidf_vectorizer': tfidf_vectorizer, 'all_genres': all_genres
    }


def get_hybrid_recommendations_for_new_user(profile_movies, n=10, cf_weight=0.5):
    """
    Generates hybrid recommendations for a new user based on their
    custom-built movie/genre profile.
    """
    
    all_selected_genres = []
    for movie in profile_movies:
        all_selected_genres.extend(movie['genres'])
    
    if not all_selected_genres:
        return [] 

    genre_string = " ".join(all_selected_genres)
    
    profile_vector = all_artifacts['tfidf_vectorizer'].transform([genre_string])
    
    content_scores = linear_kernel(profile_vector, all_artifacts['tfidf_matrix']).flatten()
    
    general_cf_scores = all_artifacts['general_cf_scores']
    
    content_weight = 1.0 - cf_weight
    hybrid_scores = (content_scores * content_weight) + (general_cf_scores * cf_weight)
            
    top_n_indices = np.argpartition(hybrid_scores, -n)[-n:]
    sorted_top_n = sorted(top_n_indices, key=lambda i: hybrid_scores[i], reverse=True)
    
    final_recs = []
    for movie_idx in sorted_top_n:
        title = all_artifacts['df_movies'].iloc[movie_idx]['title']
        score = hybrid_scores[movie_idx]
        final_recs.append(f"{title} (Score: {score:.4f})")
            
    return final_recs

def main():
    
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("🎬 Hybrid Movie Recommendation System")

    loaded_artifacts = load_all_models()
    
    if loaded_artifacts is None:
        st.error("FATAL ERROR: 'tfidf_vectorizer.joblib' not found.")
        st.error("Please make sure all .pkl, .npy, .joblib, and .npz files are in the same folder as app.py")
        return
    
    global all_artifacts
    all_artifacts = loaded_artifacts

    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_user_{uuid.uuid4().hex[:8]}"
    if 'profile_movies' not in st.session_state:
        st.session_state.profile_movies = []
        

    st.sidebar.success(f"Your Temporary User ID:\n{st.session_state.session_id}")
    st.sidebar.markdown("Build your taste profile by adding movies and their genres. The more you add, the better the recommendations!")
    
    st.sidebar.subheader("Add a Movie to Your Profile")
    with st.sidebar.form("movie_form", clear_on_submit=True):
        movie_name = st.text_input("Movie Name with year (e.g. : Inception (2010))")
        selected_genres = st.multiselect(
            "Select Genres",
            all_artifacts['all_genres']
        )
        submitted = st.form_submit_button("Add to Profile")
        if submitted and selected_genres:
            profile_movie = {"name": movie_name or "Untitled", "genres": selected_genres}
            st.session_state.profile_movies.append(profile_movie)
            st.sidebar.success(f"Added '{profile_movie['name']}' to your profile!")
        elif submitted:
            st.sidebar.warning("Please select at least one genre.")
            
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Taste Profile")
        
        if not st.session_state.profile_movies:
            st.info("Your profile is empty. Add movies in the sidebar to get started!")
        else:
            for i, movie in enumerate(st.session_state.profile_movies):
                with st.expander(f"**{movie['name']}**", expanded=True):
                    st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
        
        if st.session_state.profile_movies:
            if st.button("Clear My Profile"):
                st.session_state.profile_movies = []
                st.experimental_rerun()
                
    with col2:
        st.subheader("Your Top 20 Recommendations")
        
        if not st.session_state.profile_movies:
            st.warning("Your profile is empty. Showing you the most popular movies right now!")
            
            with st.spinner("Finding popular movies..."):
                
                popular_list = all_artifacts['popular_movies_list']
                
                random.shuffle(popular_list)
                

                random_selection = popular_list[:15]

                for i, title in enumerate(random_selection):
                    st.markdown(f"{i+1}. {title}")
                    
        else:
            
            st.success(f"Generating hybrid recommendations based on your {len(st.session_state.profile_movies)} profile movie(s)...")
                    
            with st.spinner("Analyzing your taste and running the hybrid model..."):
                recommendations = get_hybrid_recommendations_for_new_user(
                    st.session_state.profile_movies, 
                    n=1000, 
                    cf_weight=0.5 
                )
                random.shuffle(recommendations)
                top_20_recs = recommendations[:20]
                for rec in top_20_recs:
                    st.markdown(rec)

if __name__ == "__main__":
    main()