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
    write_results("knn.csv", ks, acc, "k", "accuracy")

    # train the model against a linear SVM
    log("Running Linear SVM on the dataset...")
    acc, cs = sa.svm_lin()
    log("C: " + str(cs))
    log("Accuracy: " + str(acc))
    log("Linear SVM Finished...")
    write_results("svm_lin.csv", cs, acc, "C", "accuracy")

    # train the model against a SVM
    log("Running SVM on the dataset...")
    acc, cs = sa.svm_svc()
    log("C: " + str(cs))
    log("Accuracy: " + str(acc))
    log("SVM Finished...")
    write_results("svm_rbf.csv", cs, acc, "C", "accuracy")

    # train the model against a nuSVM
    log("Running nuSVM on the dataset...")
    acc, kernels = sa.svm_nu()
    log("Kernel: " + str(kernels))
    log("Accuracy: " + str(acc))
    log("nuSVM Finished...")
    write_results("svm_nu.csv", kernels, acc, "kernel", "accuracy")

    # train the model against a SVC
    log("Running Naive Bayes on the dataset...")
    acc, methods = sa.naive_bayes()
    log("Method: " + str(methods))
    log("Accuracy: " + str(acc))
    log("Naive Bayes Finished...")
    write_results("nb.csv", methods, acc, "method", "accuracy")

    # train the model against an MLP neural network
    log("Running MLP Neural Network on the dataset...")
    acc, alphas = sa.mlp()
    log("Alphas: " + str(alphas))
    log("Accuracy: " + str(acc))
    log("MLP Neural Network Finished...")
    write_results("mlp.csv", alphas, acc, "alpha", "accuracy")

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

# Writes the results provided in a csv file with the name
def write_results(file, x, y, lbl_x, lbl_y):
    file = io.open("results\\" + file, "w", encoding="utf8")
    file.write(lbl_x + "," + lbl_y + "\n")
    for i in range(len(x)):
        file.write(str(x[i]) + "," + str(y[i]) + "\n")
    file.close()

# Saves the results as a line plot.
def save_plot(file, x, y, lbl_x, lbl_y):
    tf = pandas.DataFrame()
    plot = tf.plot()
    fig = plot.get_figure()
    fig.savefig(file)
    return "";

# Call the main method
main()