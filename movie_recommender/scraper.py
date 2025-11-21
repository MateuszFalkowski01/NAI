import requests
import yaml
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from csv_read import load_data
except ImportError:
    print("Błąd: Nie znaleziono pliku 'csv_read.py' w katalogu skryptu.")
    sys.exit(1)

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def _load_api_key():
    key_path = os.path.join(BASE_DIR, 'api_key.txt')
    
    if not os.path.exists(key_path):
        raise RuntimeError(f"Plik z kluczem API nie został znaleziony: {key_path}")
        
    with open(key_path, 'r', encoding='utf-8') as f:
        key = f.read().strip()
        
    if not key:
        raise RuntimeError("Plik 'api_key.txt' jest pusty.")
    return key

API_KEY = _load_api_key()

def scrape_movie_info(movie_title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    search_params = {
        'api_key': API_KEY,
        'query': movie_title,
        'language': 'pl-PL'
    }
    
    try:
        search_response = requests.get(search_url, params=search_params).json()
        
        if not search_response.get('results'):
            print(f"Nie znaleziono filmu: {movie_title}")
            return None
        
        movie_id = search_response['results'][0]['id']
        
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_params = {
            'api_key': API_KEY,
            'language': 'pl-PL'
        }
        
        details_response = requests.get(details_url, params=details_params).json()
        
        title_pl = details_response.get('title')
        title_org = details_response.get('original_title', 'Brak tytułu')
        overview = details_response.get('overview', 'Brak opisu')
        genres = [g['name'] for g in details_response.get('genres', [])]
        poster_path_tmdb = details_response.get('poster_path')
        release_date = details_response.get('release_date', 'Nieznana')
        final_poster_path = poster_path_tmdb
        
        info = {
            'title_pl': title_pl,
            'title_org': title_org,
            'overview': overview,
            'genres': genres,
            'release_date': release_date,
            'poster_path': final_poster_path
        }
        
        if poster_path_tmdb:
            poster_url = BASE_IMAGE_URL + poster_path_tmdb
            poster_response = requests.get(poster_url)
            if poster_response.status_code == 200:
                assets_dir = os.path.join(BASE_DIR, 'assets')
                os.makedirs(assets_dir, exist_ok=True)
                safe_title = movie_title.replace('/', '_').replace('\\', '_')
                filename = f"{safe_title}.jpg"
                file_path = os.path.join(assets_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(poster_response.content)
                info['poster_path'] = f"assets/{filename}"
                
                print(f"Zapisano poster dla {movie_title} w {file_path}")
        
        return info   
    except Exception as e:
        print(f"Błąd dla {movie_title}: {e}")
        return None

def main():
    df = load_data()
    unique_movies = df['movie'].unique()
    
    movie_data = {}
    for movie in unique_movies:
        print(f"Scraping {movie}...")
        info = scrape_movie_info(movie)
        if info:
            movie_data[movie] = info
        else:
            movie_data[movie] = {
                'title_pl': movie,
                'title_org': 'Nieznany',
                'overview': 'Brak info',
                'genres': [],
                'release_date': 'Nieznana',
                'poster_path': None
            }
    yaml_path = os.path.join(BASE_DIR, 'movie_data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(movie_data, f, allow_unicode=True, default_flow_style=False)
    print(f"Dane zapisano w: {yaml_path}")

if __name__ == "__main__":
    main()