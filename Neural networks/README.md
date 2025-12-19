# Sieci neuronowe

## Opis problemu
Zadanie polegało na budowie sieci neuronowych dostosowanych do różnych zbiorów za pomocą wybranego frameworka (tensorflow).

## Autorzy
Krzysztof Cieślik s27115
Mateusz Falkowski s27426

## Instrukcja użycia
1. Zainstaluj wymagania: `pip install -r requirements.txt`
2. Uruchom program z parametrami przykładu do predykcji np:
python main.py --model 0 --wine_params 6.2,0.66,0.48,1.2,0.029,29,75,0.9892,3.33,0.39,12.8  
python main.py --model 1 --path .\bird.jpg

## Confusion matrix
Dla modelu rozpoznającego ubrania wykonano macierz pomyłek. Jest dostępna w konsoli jak i generowana za pomocą matplotlib, przykład dostępny w pliku Confusion_matrix.png.
Przykładowy wynik w konsoli:
Macierz Pomyłek (Confusion Matrix):
[[861   1  18  27   8   1  69   0  14   1]
 [  2 974   2  16   2   0   3   0   1   0]
 [ 14   0 817   7  96   0  60   0   6   0]
 [ 17   8  11 880  34   0  44   0   5   1]
 [  1   1  36  28 873   0  55   0   6   0]
 [  0   0   0   1   0 976   0  18   1   4]
 [142   0  86  22 115   0 609   0  26   0]
 [  0   0   0   0   0  10   0 978   0  12]
 [  1   0   4   1   3   2   3   3 983   0]
 [  0   0   0   0   0   8   0  54   2 936]]

## Porównianie dwóch rozmiarów sieci neuronowej
Dla modelu rozpoznającego zwierzęta zastosowano 2 rozmiary sieci neuronowej.
Porównianie generowane jest za pomocą matplotlib, przykład dostępny w pliku Comparison.png

## Przykładowe wywołania
Przykładowe wywołania scenariuszy znajdują się w plikach Example_use1.png i Example_use2.png
