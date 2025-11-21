import csv
import pandas as pd

def load_data(file_path='data.csv'):
    """
    Load movie rating data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing user-movie-rating data.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'user', 'movie', 'rating'.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            user = row[0]
            for i in range(1, len(row), 2):
                movie = row[i]
                rating = int(row[i+1])
                data.append({'user': user, 'movie': movie, 'rating': rating})
    df = pd.DataFrame(data)
    return df

def create_user_item_matrix(df):
    """
    Create a user-item matrix from the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with 'user', 'movie', 'rating'.
    
    Returns:
        pd.DataFrame: Pivot table with users as rows, movies as columns, ratings as values.
    """
    user_item_matrix = df.pivot_table(values='rating', index='user', columns='movie', fill_value=0)
    return user_item_matrix

if __name__ == "__main__":
    df = load_data()
    print("Pierwsze 10 wierszy DataFrame:")
    print(df.head(10))
    print("\nOpis danych:")
    print(df.describe())
    print(f"\nLiczba unikalnych użytkowników: {df['user'].nunique()}")
    print(f"Liczba unikalnych filmów: {df['movie'].nunique()}")
    print(f"Całkowita liczba ocen: {len(df)}")
    
    user_item_matrix = create_user_item_matrix(df)
    print(f"\nMacierz user-item: {user_item_matrix.shape[0]} użytkowników x {user_item_matrix.shape[1]} filmów")
    print("Przykładowe wiersze macierzy:")
    print(user_item_matrix.head())
