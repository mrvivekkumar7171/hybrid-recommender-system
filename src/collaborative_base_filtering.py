from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import dask.dataframe as dd
import pandas as pd
import numpy as np


def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file
    """
    data.to_csv(file_path, index=False)


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids list and save the filtered data to a csv file
    """
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    filtered_data.loc[:, "name"] = filtered_data["name"].str.lower()
    filtered_data.loc[:, "artist"] = filtered_data["artist"].str.lower()
    # sort the data by track id
    filtered_data = filtered_data.sort_values(by="track_id")
    filtered_data.reset_index(drop=True, inplace=True)
    save_pandas_data_to_csv(filtered_data, save_df_path)


def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    save_npz(file_path, matrix)


def create_interaction_matrix(history_data:dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    """
    Create the interaction matrix from the user listening history data and save the matrix to a npz file
    """

    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    df = df.categorize(columns=['user_id', 'track_id'])
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    track_ids = df['track_id'].cat.categories.values
    np.save(track_ids_save_path, track_ids, allow_pickle=True)

    df = df.assign(user_idx=user_mapping, track_idx=track_mapping)
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()

    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()

    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    save_sparse_matrix(interaction_matrix, save_matrix_path)


def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    """
    Generate collaborative filtering recommendations based on song and artist names.
    """
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]

    input_track_id = song_row['track_id'].values.item()
    ind = np.where(track_ids == input_track_id)[0].item()
    input_array = interaction_matrix[ind]

    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_track_ids = track_ids[recommendation_indices]
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]

    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(), "score":top_scores})
    return (
        songs_data
        .loc[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df,on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )


def main(songs_data_path, user_listening_history_data_path, track_ids_save_path, filtered_data_save_path, interaction_matrix_save_path):
    songs_data = pd.read_csv(songs_data_path)
    user_data = dd.read_csv(user_listening_history_data_path)

    unique_track_ids = user_data.loc[:,"track_id"].unique().compute().tolist()

    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)


if __name__ == "__main__":
    track_ids_save_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/track_ids.npy"
    songs_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/cleaned_data.csv"
    interaction_matrix_save_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/interaction_matrix.npz"
    filtered_data_save_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/collab_filtered_data.csv"
    user_listening_history_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/raw/User Listening History.csv"

    main(songs_data_path, user_listening_history_data_path, track_ids_save_path, filtered_data_save_path, interaction_matrix_save_path)
