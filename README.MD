# Analysis of time series in the spectrum of independent components

The program aims to demonstrate the use of the FastICA algorithm on EEG data is to perform blind source separation of cortical brain activity.

## Used libraries

* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [matplot-lib](https://matplotlib.org/)

## Use cases
``python main.py [-h] -i INPUT -of OUT_FILE_PATH [-oc OUT_CHART_PATH] [--header HEADER] -n N -m M [-f {logcosh,exp,cube}] [-c CHART]``

* `-h, --help` Shows help for script  


* ```-i INPUT, --input INPUT  ``` Path to input file  


* ```-of OUT_FILE_PATH, --out-file-path OUT_FILE_PATH``` Output file path. If exist will be overwritten.


* ```-oc OUT_CHART_PATH, --out-chart-path OUT_CHART_PATH``` Output directory path for charts


* ```--header HEADER``` Bool flag if file has headers.


* `-n N`  Number of columns in input file.


* `-m M` Number of rows in input file. (Not counting the header if it exists)  


* `-f {logcosh,exp,cube}, --fun {logcosh,exp,cube} ` 
                       The functional form of the G function used in the approximation to neg-entropy

* `-c CHART, --chart CHART`
                        Bool flag that decides to save chart to out char path specified  in `-oc` (`--out-chart-path`)

### Example

`python fastEEG.py -i resources/data2x100.csv  -oc resources/ -n 2 -m 100 -f logcosh -c True -of resources/data2x100_out.csv`

####Before the execution of the algorithm.  
Signal mixture - distribution of values 
![](doc/pictures/Before2_data2x100.png)  
####After executing the algorithm
We obtain separated sources.
![](doc/pictures/After2_data2x100.png)
