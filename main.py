# import libraries and initialize them
import initialize
initialize.setup_paths()
import io
import pandas
import sentiment_analyzer
import sys
import xlrd

'''
    The entry method for the application that runs the analyzer.
'''
def main():
    # clear the log file for Main
    io.open("logs\\Main.py.log", "w").close()

    # setup the sentiment analyzer object
    sa = sentiment_analyzer.SentimentAnalyzer()

    # read the excel datasheet
    log("Reading the data from the worksheet.")
    ds = pandas.read_excel("data\\dataset.xlsx")
    i = ds["reply_id"].values
    x = ds["full_text"].values
    y = ds["label"].values

    # check if the length of the data does not match
    if len(x) != len(i) or len(y) != len(i):
        log("Error in the data sheet. Inconsitent length of rows.")
        exit(0)

    # fit the model
    log("Fitting the text collection.")
    features = sa.fit(x)

    log("Finished...")


# Logs the provided message to the 
def log(message):
    print(message + "\n")
    file = io.open("logs\\Main.py.log", "a+", encoding="utf8")
    file.write(message + "\n")
    file.close()

# Call the main method
main()