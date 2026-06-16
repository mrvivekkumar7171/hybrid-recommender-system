from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_cleaning import data_for_content_filtering
from category_encoders.count import CountEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import save_npz
import pandas as pd
import numpy as np
import joblib


def train_transformer(data, transformer_save_path):
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
        - Frequency Encoding using CountEncoder on specified columns.
        - One-Hot Encoding using OneHotEncoder on specified columns.
        - TF-IDF Vectorization using TfidfVectorizer on a specified column.
        - Standard Scaling using StandardScaler on specified columns.
        - Min-Max Scaling using MinMaxScaler on specified columns.

    Parameters:
        data (pd.DataFrame): The input data to be transformed.
        transformer_save_path (str): The file path where the trained transformer will be saved.
    Returns:
        None
    Saves:
        transformer.joblib: The trained ColumnTransformer object.
    """

    tfidf_col = 'tags'
    frequency_encode_cols = ['year']
    ohe_cols = ['artist', "time_signature", "key"]
    standard_scale_cols = ["duration_ms", "loudness", "tempo"]
    min_max_scale_cols = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]

    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ], remainder='passthrough', n_jobs=-1)
    transformer.fit(data)
    joblib.dump(transformer, transformer_save_path)
    

def transform_data(data, transformer_save_path):
    """
    Transforms the input data using a pre-trained transformer.

    Parameters:
        data (array-like): The data to be transformed.
        transformer_save_path (str): The file path where the trained transformer is saved.
    Returns:
        array-like: The transformed data.
    """

    transformer = joblib.load(transformer_save_path)
    return transformer.transform(data)


def save_transformed_data(transformed_data, transformed_data_save_path):
    """
    Save the transformed sparse matrix dataset to a specified file path.

    Parameters:
        transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
        transformed_data_save_path (str): The file path where the transformed data will be saved.
    Returns:
        None
    """

    # save sparse matrix dataset
    save_npz(transformed_data_save_path, transformed_data)


def calculate_similarity_scores(input_vector, transformed_data):
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.

    Parameters:
        input_vector (array-like): The input vector for which similarity scores are to be calculated.
        transformed_data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """

    return cosine_similarity(input_vector, transformed_data)


def content_recommendations(song_name, artist_name, songs_data, transformed_data, k=10):
    """
    lower case song and artist names, calculate similarity score and recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
        song_name (str): The name of the song to base the recommendations on.
        artist_name (str): The name of the artist of the song.
        songs_data (DataFrame): The DataFrame containing song information.
        transformed_data (ndarray): The transformed data matrix for similarity calculations.
        k (int, optional): The number of similar songs to recommend. Default is 10.
    Returns:
        DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """

    song_name = song_name.lower()
    artist_name = artist_name.lower()
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    input_vector = transformed_data[song_row.index[0]].reshape(1, -1)
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    top_k_list = top_k_songs_names[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
    return top_k_list


def main(cleaned_data_path, transformed_data_save_path, transformer_save_path):
    """
    Clean and transform the data

    Parameters:
        cleaned_data_path (str): The path to the CSV file containing the cleaned song data.
        transformed_data_save_path (str): The path where the transformed data will be saved.
        transformer_save_path (str): The path where the trained transformer will be saved.
    Returns:
        None: Prints the top k recommended songs based on content similarity.
    """

    songs_cleaned_data = pd.read_csv(cleaned_data_path)
    data_content_filtering = data_for_content_filtering(songs_cleaned_data)
    train_transformer(data_content_filtering, transformer_save_path)
    transformed_data = transform_data(data_content_filtering, transformer_save_path)
    save_transformed_data(transformed_data, transformed_data_save_path)



if __name__ == "__main__":
    TRANSFORMER_SAVE_PATH = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformer.joblib"
    CLEANED_DATA_PATH = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/cleaned_data.csv"
    TRANSFORMED_DATA_SAVE_PATH = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_data.npz"

    main(CLEANED_DATA_PATH, TRANSFORMED_DATA_SAVE_PATH, TRANSFORMER_SAVE_PATH)