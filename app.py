import streamlit as st
import pandas as pd
import pickle

# Load pre-computed cosine similarity matrix and anime data
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
anime_list = pickle.load(open('anime.pkl', 'rb'))
anime = pd.DataFrame(anime_list)

# Function to recommend anime based on title
def recommend_anime(title):
    # Normalize the input title to lowercase
    title = title.lower()

    # Find the index of the anime that matches the title (case insensitive)
    idx = anime.index[anime['title'].str.lower() == title].tolist()

    # Check if any index was found
    if not idx:
        return f"No recommendations found for '{title}'. Please check the title."

    idx = idx[0]  # Get the first matching index

    # Get pairwise similarity scores of all anime with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the anime based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get scores of the top 10 most similar anime (excluding itself)
    sim_scores = sim_scores[1:11]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar anime titles
    return anime['title'].iloc[anime_indices].tolist()

# Streamlit app layout
st.title("Anime Recommender System")

# Dropdown for selecting an anime title
option = st.selectbox('Search Anime...', anime['title'].values)

# Button to trigger recommendations
if st.button("Recommend"):
    recommendations = recommend_anime(option)
    
    # Display recommendations
    if isinstance(recommendations, str):  # If there's an error message
        st.write(recommendations)
    else:
        for idx, rec in enumerate(recommendations, start=1):
            st.write(f"{rec}")
