# Klasyfikacja z użyciem Drzew Decyzyjnych i SVM

## Opis problemu
Projekt implementuje klasyfikację danych medycznych (cukrzyca) i jakości produktów (wino) przy użyciu drzew decyzyjnych i maszyn wektorów nośnych (SVM). Opiera się na materiale "Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes".

## Autorzy
Krzysztof Cieślik s27115
Mateusz Falkowski s27426

## Instrukcja użycia
1. Zainstaluj wymagania: `pip install -r requirements.txt`
2. Uruchom program z parametrami przykładu do predykcji np:
`python main.py --w -c svm -i 7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8`
`python main.py --d -c dt -i 6,148,72,35,0,33.6,0.627,50`

## Problemy z danymi i poprawki
W zbiorze Pima Indians Diabetes Dataset występują nieprawidłowe wartości 0 w kolumnach BloodPressure, SkinThickness, Insulin, BMI i Glucose, które oznaczają brak danych (np. ciśnienie krwi równe 0 jest niemożliwe). Aby poprawić jakość danych, wartości 0 zostały zastąpione medianą odpowiedniej kolumny w funkcji `load_data`.

To eliminuje pionowe linie punktów na 0 w wizualizacjach i poprawia dokładność modeli.

Dla SVM zastosowano skalowanie cech (StandardScaler), ponieważ SVM opiera się na odległościach geometrycznych i wymaga znormalizowanych danych dla poprawnej pracy.

## Wyniki
- Metryki klasyfikacji dla obu modeli i zbiorów.
- Wizualizacje: Histogramy dla cechy Glucose (diabetes) i alcohol (wine) z podziałem na klasy, zapisane jako PNG.
- Przykładowe predykcje.

## Podsumowanie kernel functions
- Linear: Dobry dla liniowych danych.
- Poly: Dla nieliniowych, stopień zwiększa złożoność.
- RBF: Uniwersalny, gamma wpływa na czułość.
- Sigmoid: Niestabilny, podobny do NN.

## Zrzut ekranu
(Załączony w repo: screenshot.png lub wideo)
