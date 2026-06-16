from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))


from src.hybrid_recommendations import HybridRecommenderSystem
from scipy.sparse import load_npz
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Spotify Song Recommender 🚀",
    page_icon="🎶",
    layout="centered"
)


track_ids_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/track_ids.npy"
interaction_matrix_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/interaction_matrix.npz"
filtered_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/collab_filtered_data.csv"
transformed_hybrid_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_hybrid_data.npz"


track_ids = np.load(track_ids_path, allow_pickle=True)
filtered_data = pd.read_csv(filtered_data_path)
interaction_matrix = load_npz(interaction_matrix_path)
transformed_hybrid_data = load_npz(transformed_hybrid_data_path)


st.write('### Enter the name of a song and artist, the recommender will suggest similar songs 🎵🎧')


song_name = st.text_input('Enter a song name: ', value='Love Story').lower()
artist_name = st.text_input('Enter the artist name: ', value='Taylor Swift').lower()


k = st.slider(label="How many recommendations do you want ?",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1)
diversity = st.slider(label="Diversity in Recommendations",
                    min_value=1,
                    max_value=9,
                    value=5,
                    step=1)
content_based_weight = 1 - (diversity / 10)


if st.button('Get Recommendations'):
    if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
        st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")

        recommender = HybridRecommenderSystem(
            number_of_recommendations= k,
            weight_content_based= content_based_weight
            )
        recommendations = recommender.give_recommendations(
            song_name= song_name,
            artist_name= artist_name,
            songs_data= filtered_data,
            transformed_matrix= transformed_hybrid_data,
            track_ids= track_ids,
            interaction_matrix= interaction_matrix
            )

        for ind , recommendation in recommendations.iterrows():
            song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
    else:
        st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")