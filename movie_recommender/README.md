# Movie Recommender System

A movie recommendation engine using K-means clustering based on movie genres. This academic project implements content-based clustering to recommend movies similar to user's preferences.

## Description

The system uses collaborative filtering to recommend movies by finding users with similar tastes and predicting ratings for unrated movies. It features a modern GUI built with tkinter and ttkbootstrap, displaying recommendations with movie posters, descriptions, and genres.

## Algorithm

The recommendation algorithm uses K-means clustering to group movies based on their genres. 

1. **Data Loading**: Movie metadata is loaded from YAML, user ratings from CSV, and user-item matrix is created.
2. **Feature Extraction**: Movies are represented by one-hot encoded genre vectors.
3. **Clustering**: K-means algorithm groups movies into 5 clusters based on genres.
4. **Recommendation**: For a user, identify movies rated above 3 (liked). Find the most common cluster among these movies. Recommend up to 5 unwatched movies from that cluster.
5. **Anti-recommendation**: Recommend up to 5 unwatched movies from clusters different from the user's liked clusters.

## Requirements

- Python 3.8+
- Virtual environment (venv)

## Installation

1. Clone the repository.
2. Create a virtual environment:
   - **Linux/Mac**: `python3 -m venv venv`
   - **Windows**: `python -m venv venv`
3. Activate the virtual environment:
   - **Linux/Mac**: `source venv/bin/activate`
   - **Windows**: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

### Running the Application

Run `python main.py` to launch the GUI application, where you can select a user and generate movie recommendations.

### Terminal Version

You can also run the recommender from terminal: `python recommender.py`


## Project Structure

- `main.py`: Application entry point
- `csv_read.py`: Data loading and user-item matrix creation
- `recommender.py`: Recommendation algorithm implementation
- `gui/gui.py`: GUI application
- `data.csv`: User-movie rating data
- `movie_data.yaml`: Movie metadata
- `assets/`: Movie poster images
- `requirements.txt`: Python dependencies

Krzysztof Cieślik s27115 Mateusz Falkowski s27426