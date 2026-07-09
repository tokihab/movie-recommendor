import streamlit as st
import joblib
import pandas as pd
import difflib
import os

# --- MUST BE THE VERY FIRST COMMAND ---
st.set_page_config(page_title="Movie Analyzer", page_icon="🍿")

# --- Load Models with Caching ---
@st.cache_resource
def load_models():
    # Get the absolute path to the project root (assuming gui.py is in the root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    sentiment_model = joblib.load(os.path.join(models_dir, 'sentiment_model.pkl'))
    vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
    dataset = joblib.load(os.path.join(models_dir, 'movies_data.pkl'))
    similarity = joblib.load(os.path.join(models_dir, 'recommender_model.pkl'))
    
    return sentiment_model, vectorizer, dataset, similarity

sentiment_model, vectorizer, dataset, similarity = load_models()

# --- Predict Sentiment ---
def predict_sentiment(review):
    vec = vectorizer.transform([review])
    return sentiment_model.predict(vec)[0]

# --- Get Similar Movies ---
def get_similar_movies(movie_title):
    names_list = dataset['names'].tolist()
    movie_title_lower = movie_title.lower()
    close_match = None

    # 1. First, look for a substring match (Fixes the "Harry Potter" issue)
    for name in names_list:
        if movie_title_lower in str(name).lower():
            close_match = name
            break
            
    # 2. If no substring is found, fall back to difflib for typos 
    if not close_match:
        matches = difflib.get_close_matches(movie_title, names_list, n=1, cutoff=0.6)
        if not matches:
            return None, []
        close_match = matches[0]

    # Find the index of the matched movie
    idx = dataset[dataset['names'] == close_match].index[0]
    
    # Calculate similarity scores
    sim_scores = list(enumerate(similarity[idx]))
    sorted_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_similar:
        name = dataset.iloc[i[0]]['names']
        
        # Ignore the searched movie itself, and prevent duplicates
        if name != close_match and name not in recommendations:
            recommendations.append(name)
            
        if len(recommendations) >= 5:
            break
            
    return close_match, recommendations

# --- Streamlit GUI Setup ---
st.title("🎬 Movie Review Analyzer & Recommender")

movie_name = st.text_input("Enter Movie Name:")
review = st.text_area("Write Your Review:")

if st.button("Analyze Review & Get Recommendations"):
    if not movie_name or not review:
        st.warning("Please enter both a movie name and a review.")
    else:
        sentiment = predict_sentiment(review)
        match, recommendations = get_similar_movies(movie_name)
        
        if not match:
            st.error("No matching movie found in the database. Try another title.")
        else:
            # Show the actual matched movie so blind guesses are obvious
            if sentiment == 1:
                st.success(f"Positive Review! Since you liked **{match}** (matched from '{movie_name}'), check these out:")
            else:
                st.error(f"Negative Review. You didn't like **{match}** (matched from '{movie_name}'), but you might prefer:")
                
            for rec in recommendations:
                st.markdown(f"- {rec}")
