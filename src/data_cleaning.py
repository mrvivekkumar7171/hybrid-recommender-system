import pandas as pd

def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:
        1. Removes duplicate rows based on the 'spotify_id' column.
        2. Drops the 'genre' and 'spotify_id' columns.
        3. Fills missing values in the 'tags' column with the string 'no_tags'.
        4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be cleaned.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    return (
        data
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags":"no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data):
    """
    Cleans the input DataFrame by dropping specific columns. This function takes a DataFrame
    and removes the columns "track_id", "name", and "spotify_preview_url". It is intended to
    prepare the data for content based filtering by removing unnecessary features.

    Parameters:
        data (pandas.DataFrame): The input DataFrame containing songs information.
    Returns:
        pandas.DataFrame: A DataFrame with the specified columns removed.
    """

    return (
        data
        .drop(columns=["track_id", "name", "spotify_preview_url"])
    )


def main(data_path, output_path):
    """
    Main function to load, clean, and save data.

    Parameters:
        data_path (str): The file path to the raw data CSV file.
        output_path (str): The file path where the cleaned data will be saved.
    Returns:
        None
    """

    data = pd.read_csv(data_path)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    DATA_PATH = "A:/CODES/PROJECTS/hybrid-recommender-system/data/raw/Music Info.csv"
    OUTPUT_PATH = "A:/CODES/PROJECTS/hybrid-recommender-system/data/processed/cleaned_data.csv"
    main(DATA_PATH, OUTPUT_PATH)