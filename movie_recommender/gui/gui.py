import ttkbootstrap as tb
import tkinter as tk
from PIL import Image, ImageTk
import sys
import os
import yaml

class ScrollableFrame(tb.Frame):
    """
    A scrollable frame widget using canvas and scrollbar.
    
    Args:
        container: Parent widget.
        *args, **kwargs: Additional arguments for Frame.
    """
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = tb.Scrollbar(self, orient="vertical", command=self.canvas.yview, bootstyle="primary")
        self.scrollable_frame = tb.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind('<Enter>', self._bind_to_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_from_mousewheel)

    def _bind_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from csv_read import load_data, create_user_item_matrix
from recommender import recommend_movies, anti_recommend_movies, get_clustering_data

data_path = os.path.join(os.path.dirname(__file__), '..', 'data.csv')
df = load_data(data_path)
user_item_matrix = create_user_item_matrix(df)
users = list(user_item_matrix.index)

movie_data_path = os.path.join(os.path.dirname(__file__), '..', 'movie_data.yaml')
with open(movie_data_path, 'r', encoding='utf-8') as f:
    movie_data = yaml.safe_load(f)

clusters, movies_list = get_clustering_data()

class GUI:
    """
    Main GUI class for the movie recommender application.
    
    Args:
        root: Root tkinter window.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Silnik Rekomendacji Filmów")
        self.root.geometry("1400x900")

        tb.Label(root, text="Wybierz użytkownika:").pack(pady=10)
        self.user_combo = tb.Combobox(root, values=users, bootstyle="primary")
        self.user_combo.pack(pady=5)
        if users:
            self.user_combo.current(0)

        self.btn = tb.Button(root, text="Generuj Rekomendacje", command=self.get_recommendations, bootstyle="success")
        self.btn.pack(pady=10)

        self.results_frame = tb.Frame(root)
        self.results_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        self.rec_scroll = ScrollableFrame(self.results_frame)
        self.rec_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.rec_container = self.rec_scroll.scrollable_frame
        tb.Label(self.rec_container, text="Polecane filmy", font=("Arial", 16, "bold")).pack(pady=10)

        self.anti_rec_scroll = ScrollableFrame(self.results_frame)
        self.anti_rec_scroll.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        self.anti_rec_container = self.anti_rec_scroll.scrollable_frame
        tb.Label(self.anti_rec_container, text="Nie polecane filmy", font=("Arial", 16, "bold")).pack(pady=10)

    def get_recommendations(self):
        """
        Generate and display movie recommendations and anti-recommendations.
        """
        user_name = self.user_combo.get()
        if not user_name:
            return

        rec_list = recommend_movies(user_item_matrix, user_name, clusters, movies_list, top_n=5)
        anti_rec_list = anti_recommend_movies(user_item_matrix, user_name, clusters, movies_list, top_n=5)

        for widget in self.rec_container.winfo_children()[1:]:
            widget.destroy()
        for widget in self.anti_rec_container.winfo_children()[1:]:
            widget.destroy()

        for movie in rec_list:
            movie_info = movie_data.get(movie, {'title_pl': movie, 'overview': 'Brak info', 'release_date': 'Nieznana', 'genres': []})
            frame = tb.Frame(self.rec_container)
            frame.pack(pady=10, fill=tk.X)

            assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
            img_name = movie.replace('/', '_').replace(' ', '_').lower() + '.jpg'
            img_path = os.path.join(assets_dir, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((150, 225), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = tb.Label(frame, image=photo)
                img_label.image = photo
                img_label.pack(side=tk.LEFT, padx=10)
            else:
                tb.Label(frame, text="Brak obrazu").pack(side=tk.LEFT, padx=10)

            text_frame = tb.Frame(frame)
            text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            title_text = tb.Text(text_frame, height=1, font=("Arial", 12, "bold"), wrap=tk.WORD)
            title_text.insert(tk.END, movie_info.get('title_pl', movie))
            title_text.config(state='disabled')
            title_text.pack(anchor=tk.W, fill=tk.X)
            overview_text = tb.Text(text_frame, height=3, wrap=tk.WORD)
            overview_text.insert(tk.END, movie_info.get('overview', ''))
            overview_text.config(state='disabled')
            overview_text.pack(anchor=tk.W, fill=tk.X)
            genres_text = tb.Text(text_frame, height=1, wrap=tk.WORD)
            genres_text.insert(tk.END, f"Gatunki: {', '.join(movie_info.get('genres', []))}")
            genres_text.config(state='disabled')
            genres_text.pack(anchor=tk.W, fill=tk.X)
            date_text = tb.Text(text_frame, height=1, wrap=tk.WORD)
            date_text.insert(tk.END, movie_info.get('release_date', ''))
            date_text.config(state='disabled')
            date_text.pack(anchor=tk.W, fill=tk.X)

        if not rec_list:
            tb.Label(self.rec_container, text="Brak rekomendacji.").pack(pady=10)

        for movie in anti_rec_list:
            movie_info = movie_data.get(movie, {'title_pl': movie, 'overview': 'Brak info', 'release_date': 'Nieznana', 'genres': []})
            frame = tb.Frame(self.anti_rec_container)
            frame.pack(pady=10, fill=tk.X)

            assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
            img_name = movie.replace('/', '_').replace(' ', '_').lower() + '.jpg'
            img_path = os.path.join(assets_dir, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((150, 225), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = tb.Label(frame, image=photo)
                img_label.image = photo
                img_label.pack(side=tk.LEFT, padx=10)
            else:
                tb.Label(frame, text="Brak obrazu").pack(side=tk.LEFT, padx=10)

            text_frame = tb.Frame(frame)
            text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            title_text = tb.Text(text_frame, height=1, font=("Arial", 12, "bold"), wrap=tk.WORD)
            title_text.insert(tk.END, movie_info.get('title_pl', movie))
            title_text.config(state='disabled')
            title_text.pack(anchor=tk.W, fill=tk.X)
            overview_text = tb.Text(text_frame, height=3, wrap=tk.WORD)
            overview_text.insert(tk.END, movie_info.get('overview', ''))
            overview_text.config(state='disabled')
            overview_text.pack(anchor=tk.W, fill=tk.X)
            genres_text = tb.Text(text_frame, height=1, wrap=tk.WORD)
            genres_text.insert(tk.END, f"Gatunki: {', '.join(movie_info.get('genres', []))}")
            genres_text.config(state='disabled')
            genres_text.pack(anchor=tk.W, fill=tk.X)
            date_text = tb.Text(text_frame, height=1, wrap=tk.WORD)
            date_text.insert(tk.END, movie_info.get('release_date', ''))
            date_text.config(state='disabled')
            date_text.pack(anchor=tk.W, fill=tk.X)

        if not anti_rec_list:
            tb.Label(self.anti_rec_container, text="Brak filmów do anty-rekomendacji.").pack(pady=10)

def main():
    """
    Main entry point for the GUI application.
    """
    root = tb.Window(themename="darkly")
    gui = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()