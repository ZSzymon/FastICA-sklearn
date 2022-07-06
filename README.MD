# Analiza szeregów czasowych w spektrum niezależnych składowych

Program na na celu zaprezentować użycie algorytmu FastICA na danych EEG ma na celu wykonać slepą separacje źródeł aktywności korowej mózgu.

## Wykorzystane bibloteki

* ###[scikit-learn](https://scikit-learn.org/stable/index.html)
* ###[matplot-lib](https://matplotlib.org/)

## Instrukcja użycia
``python main.py [-h] -i INPUT -of OUT_FILE_PATH [-oc OUT_CHART_PATH] [--header HEADER] -n N -m M [-f {logcosh,exp,cube}] [-c CHART]``

`-h, --help`pokazuje intrukcje obsługi w języku angielskim

```-i INPUT, --input INPUT  ``` ścieżka do pliku wejściowego

  ```-of OUT_FILE_PATH, --out-file-path OUT_FILE_PATH``` output file path. If exist will be overwritten.

  ```-oc OUT_CHART_PATH, --out-chart-path OUT_CHART_PATH``` output directory path for charts

  ```--header HEADER``` Bool flag if file has headers.

  `-n N`  liczba kolumn w pliku źródłowym

  `-m M` liczba wierszy w pliku źródłowym (nie licząc nagłówka jeżeli istnieje)  

  `-f {logcosh,exp,cube}, --fun {logcosh,exp,cube} ` 
                        The functional form of the G function used in the approximation to neg-entropy

  `-c CHART, --chart CHART`
                        Flaga bool decydująca o zapisaniu wykresów do folderu podanego w `-oc`  

### Przykład

`python fastEEG.py -i resources/data2x100.csv  -oc resources/ -n 2 -m 100 -f logcosh -c True -of resources/data2x100_out.csv`

Przed wykonaniem algorytmu  Mieszanina sygnałów – rozkład wartości 
![](doc/pictures/Before2_data2x100.png)  
Po wykonaniu algorytmu uzyskujemy odseparowane źródła. 
![](doc/pictures/After2_data2x100.png)
