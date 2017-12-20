# import libraries and initialize them
import initialize
initialize.setup_paths()
import io
import math
import pandas
import sentiment_analyzer
import sklearn.utils
import sys
import xlrd

'''
    The entry method for the application that runs the analyzer.
'''
def main():
    # clear the log file for Main
    io.open("logs\\Main.py.log", "w").close()

    # read the stopwords
    sw = pandas.read_excel("data\\stopwords.xlsx")
    stopwords = []
    for item in  sw["stopwords"].values:
        stopwords.append(item)

    # read the excel datasheet
    log("Reading the data from the worksheet.")
    ds = pandas.read_excel("data\\dataset.xlsx")

    # retrieve the list from the columns for the data
    x = get_list(ds, "data")
    y = get_list(ds, "labels")

    # check if the length of the data does not match
    if len(x) != len(y):
        log("Error in the data sheet. Inconsistent number of rows.")
        exit(0)

    # setup the sentiment analyzer object
    sa = sentiment_analyzer.SentimentAnalyzer(x, y, 0.20, stopwords)

    # train the model against a kNN
    log("Running kNN on the dataset...")
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    acc, ks = sa.knn(ks)
    log("k: " + str(ks))
    log("Accuracy: " + str(acc))
    log("kNN Finished...")

    # train the model against a linear SVM
    log("Running Linear SVM on the dataset...")
    cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    acc, cs = sa.svm_lin(cs)
    log("C: " + str(cs))
    log("Accuracy: " + str(acc))
    log("Linear SVM Finished...")

    # train the model against a SVM
    log("Running SVM on the dataset...")
    cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    acc, cs = sa.svm_svc(cs)
    log("C: " + str(cs))
    log("Accuracy: " + str(acc))
    log("SVM Finished...")

    # train the model against a nuSVM
    log("Running nuSVM on the dataset...")
    cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    acc, cs = sa.svm_nu()
    log("C: " + str(cs))
    log("Accuracy: " + str(acc))
    log("nuSVM Finished...")

    # train the model against a SVC
    log("Running Naive Bayes on the dataset...")
    acc, methods = sa.naive_bayes()
    log("Method: " + str(methods))
    log("Accuracy: " + str(acc))
    log("Naive Bayes Finished...")

# Gets the list from the excel data columns
def get_list(datasheet, column_name):
    list = []
    items = datasheet[column_name].values
    for item in items:
        if item != None and str(item) != "nan":
            list.append(item)
    return list

# Logs the provided message to the 
def log(message):
    print(message + "\n")
    file = io.open("logs\\Main.py.log", "a+", encoding="utf8")
    file.write(message + "\n")
    file.close()

# Call the main method
main()