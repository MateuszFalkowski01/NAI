"""
Opis problemu:
Projekt dotyczy analizy i klasyfikacji danych medycznych oraz jakości produktów przy użyciu drzew decyzyjnych i maszyn wektorów nośnych (SVM).
Na podstawie materiału "Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes" (https://jns.edu.al/wp-content/uploads/2024/01/M.UlqinakuA.Ktona-FINAL.pdf),
implementujemy klasyfikatory dla zbiorów danych: Pima Indians Diabetes Dataset oraz Wine Quality Dataset.

Krzysztof Cieślik s27115
Mateusz Falkowski s27426

Instrukcja użycia:
Uruchomić skrypt z parametrami do predykcji np: 
python main.py --w -c svm -i 7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8
python main.py --d -c dt -i 6,148,72,35,0,33.6,0.627,50

Referencje:
- Materiał: https://jns.edu.al/wp-content/uploads/2024/01/M.UlqinakuA.Ktona-FINAL.pdf
- Pima Indians Diabetes Dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- Wine Quality Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def load_data(filepath, sep=',', target_col=-1):
    """
    Ładuje dane z pliku CSV i czyści nieprawidłowe wartości (0 zastąpione medianą dla wybranych kolumn).

    Args:
        filepath (str): Ścieżka do pliku CSV.
        sep (str): Separator w pliku (domyślnie ',').
        target_col (int): Indeks kolumny docelowej (domyślnie ostatnia).

    Returns:
        tuple: (X, y) gdzie X to cechy, y to etykiety.
    """
    df = pd.read_csv(filepath, sep=sep)
    # Dla diabetes dataset: zastąp 0 medianą w kolumnach gdzie 0 oznacza brak danych
    if 'BloodPressure' in df.columns:
        df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
    if 'SkinThickness' in df.columns:
        df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
    if 'Insulin' in df.columns:
        df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
    if 'Glucose' in df.columns:
        df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
    X = df.iloc[:, :target_col]
    y = df.iloc[:, target_col]
    return X, y

def train_and_evaluate(X, y, model, scale=False):
    """
    Dzieli dane na train/test, trenuje model i oblicza metryki.
    Dla SVM stosuje skalowanie cech (StandardScaler).

    Args:
        X (pd.DataFrame): Cechy.
        y (pd.Series): Etykiety.
        model: Model sklearn.
        scale (bool): Czy skalować dane.

    Returns:
        tuple: (metrics, model, scaler) Metryki, wytrenowany model i scaler (None, jeśli brak skalowania).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return metrics, model, scaler

def visualize_data(X, y, title, feature='Glucose'):
    """
    Wizualizuje dane - histogram jednej cechy z podziałem na klasy.

    Args:
        X (pd.DataFrame): Cechy.
        y (pd.Series): Etykiety.
        title (str): Tytuł wykresu.
        feature (str): Nazwa cechy do wizualizacji.
    """
    df = X.copy()
    df['target'] = y
    plt.figure()
    sns.histplot(data=df, x=feature, hue='target', kde=True)
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

def predict_example(model, example_data, feature_names, scaler=None):
    """
    Przewiduje dla przykładowego wejścia.

    Args:
        model: Wytrenowany model.
        example_data (list): Przykładowe dane.
        feature_names (list): Nazwy cech.
        scaler (StandardScaler, optional): Użyty do skalowania danych treningowych.

    Returns:
        int: Przewidywana klasa.
    """
    try:
        df = pd.DataFrame([example_data], columns=feature_names)
        data_for_pred = df
        if scaler is not None:
            data_for_pred = scaler.transform(df)
            
        return model.predict(data_for_pred)[0]
    except ValueError as e:
        print(f"Błąd podczas predykcji: {e}")
        print("Sprawdź, czy liczba cech wejściowych zgadza się z oczekiwaną liczbą cech dla wybranego zestawu danych.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Klasyfikacja Diabetes i Wine Quality za pomocą Decision Tree i SVM.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--d', action='store_true', help='Używa modelu wytrenowanego na zbiorze Diabetes.')
    group.add_argument('--w', action='store_true', help='Używa modelu wytrenowanego na zbiorze Wine Quality.')
    
    parser.add_argument('-c', '--classifier', choices=['dt', 'svm'], required=True, 
                        help='Wybór klasyfikatora: dt (Decision Tree) lub svm (SVC rbf kernel).')
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Wartości cech do predykcji, oddzielone przecinkami (np. 6,148,72,35,0,33.6,0.627,50).')

    args = parser.parse_args()

    # Ładowanie i przygotowanie danych
    if args.d:
        dataset_name = "Diabetes"
        filepath = 'diabetes.csv'
        sep = ','
        X, y = load_data(filepath, sep=sep)
    elif args.w:
        dataset_name = "Wine Quality"
        filepath = 'winequality-white.csv'
        sep = ';'
        X, y = load_data(filepath, sep=sep)
        y = (y > 5).astype(int)
    else:
        # Ten kod nie powinien być osiągalny, ale jest dla pewności
        print("Nie wybrano zestawu danych (--d lub --w).")
        sys.exit(1)
        
    feature_names = X.columns
    print(f"Wybrano zbiór: **{dataset_name}**")
    print(f"Oczekiwana liczba cech: **{len(feature_names)}** ({', '.join(feature_names.tolist())})")

    # Parsowanie i walidacja danych wejściowych
    try:
        input_data = [float(x.strip()) for x in args.input.split(',')]
    except ValueError:
        print("Błąd: Dane wejściowe muszą być liczbami oddzielonymi przecinkami.")
        sys.exit(1)
        
    if len(input_data) != len(feature_names):
        print(f"Błąd: Oczekiwano {len(feature_names)} cech, podano {len(input_data)}.")
        sys.exit(1)

    # Trening wybranego modelu
    model = None
    scaler = None
    
    if args.classifier == 'dt':
        print("\nWybrano klasyfikator: **Decision Tree**")
        dt = DecisionTreeClassifier(random_state=42)
        metrics, model, _ = train_and_evaluate(X, y, dt, scale=False)
    elif args.classifier == 'svm':
        print("\nWybrano klasyfikator: **SVM (rbf kernel)**")
        svc = SVC(kernel='rbf', gamma=1, random_state=42)
        metrics, model, scaler = train_and_evaluate(X, y, svc, scale=True)
    
    print(f"Metryki wytrenowanego modelu: {metrics}")

    # Predykcja na danych z konsoli
    if model:
        prediction = predict_example(model, input_data, feature_names, scaler)
        print("\n--- **Wynik Predykcji** ---")
        print(f"Dane wejściowe: **{args.input}**")
        print(f"Przewidywana klasa: **{prediction}**")
        
        if args.d:
            print(f"Znaczenie (Diabetes): {prediction} to {'Brak cukrzycy (0)' if prediction == 0 else 'Cukrzyca (1)'}")
        elif args.w:
            print(f"Znaczenie (Wine Quality): {prediction} to {'Niska jakość (0: <=5)' if prediction == 0 else 'Wysoka jakość (1: >5)'}")
    else:
        print("Błąd: Model nie został wytrenowany.")


if __name__ == "__main__":
    main()
