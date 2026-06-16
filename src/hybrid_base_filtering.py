from src.content_based_filtering import transform_data, save_transformed_data
from src.data_cleaning import data_for_content_filtering
import pandas as pd


def main(data_path, save_path, transformer_save_path):
    """
    load, clean, transform the `collaborative filtered` dataset into spare matrix and save it

    Using the `collaborative filtered` dataset instead of `Music Info` dataset due to shape mismatch
    because `collaborative filtered` dataset contains only those songs which are present in `User Listening History` dataset
    """

    filtered_data = pd.read_csv(data_path)
    filtered_data_cleaned = data_for_content_filtering(filtered_data)
    transformed_data = transform_data(filtered_data_cleaned, transformer_save_path)
    save_transformed_data(transformed_data, save_path)


if __name__ == "__main__":
    filtered_data_path = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/collab_filtered_data.csv"
    save_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformed_hybrid_data.npz"
    transformer_save_path = "A:/CODES/PROJECTS/hybrid-recommender-system/models/transformer.joblib"

    main(filtered_data_path, save_path, transformer_save_path)