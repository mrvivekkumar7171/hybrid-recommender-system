from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))


from src.collaborative_base_filtering import collaborative_recommendation
from src.content_based_filtering import content_recommendations
from src.hybrid_recommendations import HybridRecommenderSystem
from scipy.sparse import load_npz
from numpy import load
import pandas as pd
import numpy as np


def test_content_recommendations(song_name, artist_name, songs_data, transformed_data, k):
    """
    load the cleaned data and transformed data
    test the content_recommendations function
    """

    recommendations = content_recommendations(song_name, artist_name, songs_data, transformed_data, k)

    print("\nContent-Based Recommendations:")
    for ind , recommendation in recommendations.iterrows():
        print(f"{ind}. {recommendation['name'].title()} by {recommendation['artist'].title()}")


def test_collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    """
    load the cleaned data and transformed data
    test the collaborative_recommendation function
    """

    recommendations = collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k)

    print("\nCollaborative Recommendations:")
    for ind , recommendation in recommendations.iterrows():
        print(f"{ind}. {recommendation['name'].title()} by {recommendation['artist'].title()}")


def test_hybrid_recommendation(song_name, artist_name, filtered_songs_data, transformed_hybrid_data, track_ids, interaction_matrix, weight_content_based, k=5):
    """
    load the cleaned data and transformed data
    test the hybrid_recommendation function
    """

    filtered_songs_data = filtered_songs_data[["track_id", "name", "artist", "spotify_preview_url"]]
    rec = HybridRecommenderSystem(
                                number_of_recommendations= k,
                                weight_content_based= weight_content_based
                                )
    recommendations = rec.give_recommendations(
                                            song_name= song_name,
                                            artist_name= artist_name,
                                            songs_data= filtered_songs_data,
                                            transformed_matrix= transformed_hybrid_data,
                                            track_ids= track_ids,
                                            interaction_matrix= interaction_matrix
                                            )

    print("\nHybrid Recommendations:")
    for ind , recommendation in recommendations.iterrows():
        print(f"{ind}. {recommendation['name'].title()} by {recommendation['artist'].title()}")


if __name__ == "__main__":

    songs_data = pd.read_csv("A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/cleaned_data.csv")
    transformed_data = load_npz("A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_data.npz")
    track_ids = np.load("A:/CODES/PROJECTS/hybrid-recommender-system/models/track_ids.npy", allow_pickle=True)
    filtered_songs_data = pd.read_csv("A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/collab_filtered_data.csv")
    interaction_matrix = load_npz("A:/CODES/PROJECTS/hybrid-recommender-system/models/interaction_matrix.npz")
    transformed_hybrid_data = load_npz("A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_hybrid_data.npz")

    k = 5
    weight_content_based = 0.3
    song_name = "Love Story"
    artist_name = "Taylor Swift"

    test_content_recommendations(song_name, artist_name, songs_data, transformed_data, k)
    test_collaborative_recommendation(song_name, artist_name, track_ids, filtered_songs_data, interaction_matrix, k)
    test_hybrid_recommendation(song_name, artist_name, filtered_songs_data, transformed_hybrid_data, track_ids, interaction_matrix, weight_content_based, k)