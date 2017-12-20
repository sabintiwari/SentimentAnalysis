## Sentiment Analysis for Nepali Phrases | Sabin Raj Tiwari | CMSC 678 Project

***
#### Setup and Requirements
***
* Python 3.6.3
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [morfessor](https://github.com/aalto-speech/morfessor)
* [pandas](https://github.com/pandas-dev/pandas)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [indic_nlp_library](https://github.com/anoopkunchukuttan/indic_nlp_library)
* [tweepy](https://github.com/tweepy/tweepy)
* [xlrd](https://github.com/python-excel/xlrd)

There is a `setup.bat` file that is included that can be run to install the required dependencies. This assumes that the `pip` script is available. Python 3.4+ comes default with `pip` installed although the path to the script may need to be added to the environment variables. Please run `setup.bat` with administrator privileges.

Note: This project has only been tested on Windows 10.

***
#### Run
***
The main script of the project is contained in the `main.py` file. In order to run the program, you can run it like so from command prompt/shell:
```bash
python ./main.py
```

This will create a log file in the logs folder name `Main.py.log` which will contain the outputs of the program.

It will also output the results of each of the models in a .csv file in the results folder with their respective names.

***
#### Contents
***
`data/dataset.xlsx` - The Excel sheet containing the dataset that was used along with the labels.
`data/stopwords.xlsx` - The Excel sheet containing the stopwords that gets used to transform the data.
`results/results.xlsx` - The Excel sheet that contains the results of the experiments.
`config.json` - Configuration file used to get the Twitter data. I have cleared out the keys and tokens used to query the data because those were from my Twitter dev account. If you would like to use the fetch_data.py code to retrieve Twitter data, then the keys need to be generated.
`fetch_data.py` - Script that retrieves data from Twitter using the config file. Outputs the results in a csv file.
`initialize.py` - Initialize the paths to the indic-nlp-library and the data needed by it.
`main.py` - The main script of the project that uses the dataset and the sentiment_analyzer to perform all the experiments and write the results.
`sentiment_analyzer.py` - The class that does the work of transforming the data, creating the models, and has methods that can be called to perform experiments on those models.