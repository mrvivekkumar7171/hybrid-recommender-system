"""content based filtering similarity scores are not shorted while collaborative filtering similarity scores are
sorted in ascending order so, we need to sort both the similarity scores in ascending order before combining them
this is done by sorting the filtered dataset in ascending order of track_id because we are working with index in
content based filtering while in collaborative filtering, we are fetching track_id from the track_ids list"""

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import pandas as pd
import numpy as np


class HybridRecommenderSystem:
    """
    Class to implement the Hybrid Recommender System which combines Content-Based Filtering and Collaborative Filtering techniques.
    """
    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based


    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        """
        Filter the song with given name and artist from songs dataset, get the index of the song
        generate the input vector from the transformed matrix and calculate the cosine similarity scores.
        """

        song_row = songs_data.loc[(songs_data["name"] == song_name.lower()) & (songs_data["artist"] == artist_name.lower())]
        song_index = song_row.index[0]
        input_vector = transformed_matrix[song_index].reshape(1,-1)
        return cosine_similarity(input_vector, transformed_matrix)


    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        """
        Filter the song with given name and artist from songs dataset, get the track_id of the song
        fetch the index from the track_ids list, fetch input vector from the interaction matrix
        and calculate the cosine similarity scores.
        """

        song_row = songs_data.loc[(songs_data["name"] == song_name.lower()) & (songs_data["artist"] == artist_name.lower())]
        input_track_id = song_row['track_id'].values.item()
        ind = np.where(track_ids == input_track_id)[0].item()
        input_array = interaction_matrix[ind]
        return cosine_similarity(input_array, interaction_matrix)


    def __normalize_similarities(self, similarity_scores):
        """
        Normalizes the similarity scores using min-max normalization to bring them to a common scale before combining them
        as similarity scores of content based filtering is high in comparision to collaborative filtering because collaborative
        filtering is sparse matrix
        """

        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        return (similarity_scores - minimum) / (maximum - minimum)


    def __weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        """
        Combines the normalized similarity scores from both content-based and collaborative filtering using the specified weights.
        """

        return (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)


    def give_recommendations(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        """
        Calculate Content Based and Collaborative Filtering Similarities then Normalize those Similarities
        Calculate weighted combination of those similarities and get the index values of those recommendations
        get top k recommendations, scores and print the songs from data.
        """

        content_based_similarities = self.__calculate_content_based_similarities(song_name= song_name, 
                                                                               artist_name= artist_name, 
                                                                               songs_data= songs_data, 
                                                                               transformed_matrix= transformed_matrix
                                                                               )
        collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(song_name= song_name, 
                                                                                                   artist_name= artist_name, 
                                                                                                   track_ids= track_ids, 
                                                                                                   songs_data= songs_data, 
                                                                                                   interaction_matrix= interaction_matrix
                                                                                                   )

        normalized_content_based_similarities = self.__normalize_similarities(content_based_similarities)
        normalized_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)
        weighted_scores = self.__weighted_combination(content_based_scores= normalized_content_based_similarities, 
                                                    collaborative_filtering_scores= normalized_collaborative_filtering_similarities)


        recommendation_track_ids = track_ids[np.argsort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]]
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]

        scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(), "score":top_scores})
        return (
            songs_data
            .loc[songs_data["track_id"].isin(recommendation_track_ids)]
            .merge(scores_df, on="track_id")
            .sort_values(by="score", ascending=False)
            .drop(columns=["track_id", "score"])
            .reset_index(drop=True)
        )

if __name__ == "__main__":
    transformed_hybrid_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_hybrid_data.npz"
    interaction_matrix_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/interaction_matrix.npz"
    track_ids_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/track_ids.npy"
    filtered_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/collab_filtered_data.csv"

    transformed_hybrid_data = load_npz(transformed_hybrid_data_path)
    interaction_matrix = load_npz(interaction_matrix_path)
    track_ids = np.load(track_ids_path, allow_pickle=True)
    filtered_data = pd.read_csv(filtered_data_path, usecols=["track_id", "name", "artist", "spotify_preview_url"])

    rec = HybridRecommenderSystem(
                                number_of_recommendations= 10,
                                weight_content_based= 0.3
                                )
    recommendations = rec.give_recommendations(
                                            song_name= 'Love Story',
                                            artist_name= 'Taylor Swift',
                                            songs_data= filtered_data,
                                            transformed_matrix= transformed_hybrid_data,
                                            track_ids= track_ids,
                                            interaction_matrix= interaction_matrix
                                            )
    print(recommendations)