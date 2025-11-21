from csv_read import load_data, create_user_item_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

def load_movie_data(file_path='movie_data.yaml'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def create_movie_features(movie_data):
    movies = []
    genres_list = []
    for movie, info in movie_data.items():
        if 'genres' in info and info['genres']:
            movies.append(movie)
            genres_list.append(info['genres'])
    mlb = MultiLabelBinarizer()
    features = mlb.fit_transform(genres_list)
    return movies, features, mlb.classes_

def cluster_movies(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters

def get_clustering_data():
    movie_data = load_movie_data()
    movies_list, features, genres = create_movie_features(movie_data)
    clusters = cluster_movies(features, n_clusters=5)
    return clusters, movies_list

def find_similar_users(user_item_matrix, user_name, top_n=5):
    """
    Find top similar users using cosine similarity.
    
    Args:
        user_item_matrix (pd.DataFrame): User-item matrix.
        user_name (str): Target user.
        top_n (int): Number of similar users to find.
    
    Returns:
        dict: Dictionary of similar users and their similarity scores.
    """
    if user_name not in user_item_matrix.index:
        return {}
    
    similarity_matrix = cosine_similarity(user_item_matrix)
    user_index = user_item_matrix.index.get_loc(user_name)
    similarities = similarity_matrix[user_index]
    
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    similar_users = {user_item_matrix.index[i]: similarities[i] for i in similar_indices}
    return similar_users

def predict_ratings(user_item_matrix, similar_users, user_name):
    """
    Predict ratings for movies using weighted average from similar users.
    
    Args:
        user_item_matrix (pd.DataFrame): User-item matrix.
        similar_users (dict): Similar users and scores.
        user_name (str): Target user.
    
    Returns:
        dict: Predicted ratings for movies.
    """
    predictions = {}
    for movie in user_item_matrix.columns:
        numerator = 0.0
        denominator = 0.0
        for sim_user, sim in similar_users.items():
            rating = user_item_matrix.loc[sim_user, movie]
            if rating > 0:
                numerator += rating * sim
                denominator += sim
        if denominator > 0:
            predictions[movie] = numerator / denominator
    return predictions

def recommend_movies(user_item_matrix, user_name, clusters, movies_list, top_n=5):
    """
    Recommend top n movies based on clustering of user's liked movies.
    
    Args:
        user_item_matrix (pd.DataFrame): User-item matrix.
        user_name (str): Target user.
        clusters: Cluster labels for movies.
        movies_list: List of movies in order.
        top_n (int): Number of recommendations.
    
    Returns:
        list: Top n recommended movies.
    """
    if user_name not in user_item_matrix.index:
        return []
    
    user_ratings = user_item_matrix.loc[user_name]
    watched_normalized = set(m.strip().lower() for m in user_ratings[user_ratings > 0].index)
    
    liked_movies = user_ratings[user_ratings > 7].index.tolist()
    
    if not liked_movies:
        return []
    
    # Find clusters of liked movies
    liked_clusters = []
    for movie in liked_movies:
        movie_normalized = movie.strip().lower()
        if any(m.strip().lower() == movie_normalized for m in movies_list):
            idx = next(i for i, m in enumerate(movies_list) if m.strip().lower() == movie_normalized)
            liked_clusters.append(clusters[idx])
    
    if not liked_clusters:
        return []
    
    # Most common cluster
    most_common_cluster = Counter(liked_clusters).most_common(1)[0][0]
    
    # Movies in that cluster
    cluster_movies = [movies_list[i] for i in range(len(movies_list)) if clusters[i] == most_common_cluster]
    
    # Exclude watched
    candidates = [m for m in cluster_movies if m.strip().lower() not in watched_normalized]
    
    # Sort alphabetically
    candidates.sort()
    
    # Return first top_n
    return candidates[:top_n]

def anti_recommend_movies(user_item_matrix, user_name, clusters, movies_list, top_n=5):
    """
    Anti-recommend top n movies from the cluster with the least liked movies by the user, excluding watched.
    
    Args:
        user_item_matrix (pd.DataFrame): User-item matrix.
        user_name (str): Target user.
        clusters: Cluster labels for movies.
        movies_list: List of movies in order.
        top_n (int): Number of anti-recommendations.
    
    Returns:
        list: Top n anti-recommended movies (unwatched from least liked cluster).
    """
    if user_name not in user_item_matrix.index:
        return []
    
    user_ratings = user_item_matrix.loc[user_name]
    watched_normalized = set(m.strip().lower() for m in user_ratings[user_ratings > 0].index)
    
    liked_movies = user_ratings[user_ratings > 7].index.tolist()
    liked_cluster_list = []
    for movie in liked_movies:
        movie_normalized = movie.strip().lower()
        if any(m.strip().lower() == movie_normalized for m in movies_list):
            idx = next(i for i, m in enumerate(movies_list) if m.strip().lower() == movie_normalized)
            liked_cluster_list.append(clusters[idx])
    
    if liked_cluster_list:
        liked_cluster_counts = Counter(liked_cluster_list)
        least_liked_cluster = min(liked_cluster_counts, key=liked_cluster_counts.get)
        anti_candidates = [movies_list[i] for i in range(len(movies_list)) if clusters[i] == least_liked_cluster and movies_list[i].strip().lower() not in watched_normalized]
    else:
        # If no liked movies, take all unwatched
        anti_candidates = [m for m in movies_list if m.strip().lower() not in watched_normalized]
    
    # Sort alphabetically
    anti_candidates.sort()
    
    return anti_candidates[:top_n]


if __name__ == "__main__":
    df = load_data()
    user_item_matrix = create_user_item_matrix(df)
    clusters, movies_list = get_clustering_data()
    
    print("Dostępni użytkownicy:")
    for i, user in enumerate(user_item_matrix.index):
        print(f"{i+1}. {user}")
    
    try:
        choice = int(input("Wybierz numer użytkownika: ")) - 1
        if 0 <= choice < len(user_item_matrix.index):
            selected_user = user_item_matrix.index[choice]
        else:
            print("Nieprawidłowy wybór. Wybieram pierwszego użytkownika.")
            selected_user = user_item_matrix.index[0]
    except ValueError:
        print("Nieprawidłowa wartość. Wybieram pierwszego użytkownika.")
        selected_user = user_item_matrix.index[0]
    
    print(f"\nRekomendacja dla {selected_user}:")
    rec = recommend_movies(user_item_matrix, selected_user, clusters, movies_list)
    print(rec)
    
    print(f"\nAntyrekomendacja dla {selected_user}:")
    anti_rec = anti_recommend_movies(user_item_matrix, selected_user, clusters, movies_list)
    print(anti_rec)