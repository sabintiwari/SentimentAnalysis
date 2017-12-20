# import libraries
import math
import numpy
import re
import sys

# import the indic nlp library
from indicnlp import common
from indicnlp import loader
from indicnlp.morph import unsupervised_morph
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# import the sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import utils

# Class that handles the sentiment analysis.
class SentimentAnalyzer(object):

    # Properties
    emoticons_str = r"""
                    (?: 
                        [:=;]
                        [D\)\]\(\]/\\OpP]
                    )"""

    ignore_tokens = [
        r"<[^>]+>", 
        r"(?:@[\w_]+)", 
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
        r"http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+"
    ]

    custom_tokens = {}

    ''' 
        Initialization function.
        x is the list of data.
        y is the list of labels for the data.
        ratio is the ratio is the ratio of test data compared to train data that should be used
        stopwords are words that should be ignored when creating the dtm.
    '''
    def __init__(self, x, y, ratio=0.10, stopwords=[]):
        # initialize the indic nlp library components
        loader.load()
        self.factory = IndicNormalizerFactory()
        # self.morph_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('hi')
        self.normalizer = self.factory.get_normalizer("ne", False)
        self.tokenizer = indic_tokenize

        # initialize the TfidfVectorizer using the custom tokenizer fron indic-nlp-library
        self.vectorizer = TfidfVectorizer(analyzer="word", binary=False, encoding=u'utf-8', input=u'content', 
                                          lowercase=False, tokenizer=self.tokenize_text, stop_words=stopwords)

        # shuffle the data first
        x, y = utils.shuffle(x, y, random_state=2)

        # get certain number of testing and training data
        test_size = math.floor(len(x) * ratio)
        train_size = len(x) - test_size

        # fit the data values and convert them into a sparse matrix
        dtm = self.fit_transform(x)

        # separate the training and testing data
        self.x_train = dtm[:train_size]
        self.x_test = dtm[(-1 * test_size):]
        self.y_train = y[:train_size]
        self.y_test = y[-1 * test_size:]

    '''
        Method that cleans the message by removing unncessary information using regex.
        This removes HTML tags, mentions, hashtags, links, and leading/trailing whitespaces.
    '''
    def clean_text(self, text):
        for token in self.ignore_tokens:
            text = re.sub(token, "", text)
        return text.strip()

    '''
        Calls the fit() method of the TfidfVectorizer on the collection of text values.
        Return the DTM that are learned from the TD-IDF Vectorizer.
    '''
    def fit_transform(self, text_collection):
        # normalize the text first
        messages = []
        for item in text_collection:
            text = self.normalize_text(item)
            messages.append(text)
        # call the fit method of the vectorizer
        dtm = self.vectorizer.fit_transform(messages)
        return dtm

    '''
        Gets the accuracy for the provided z and the labels.
    '''
    def get_acc(self, z, labels):
        # check if the predictions were correct
        correct = 0
        for i in range(0, len(labels)):
            if z[i] == labels[i]:
                correct = correct + 1
        # return the accuracy
        return correct / len(labels) * 100.00

    '''
        Gets the list of features that are currently learned in the vectorizer.
    '''
    def get_features(self):
        return self.vectorizer.get_feature_names()

    '''
        Trains and tests the dataset using a knn for the different number of ks provided.
        Returns the list of accuracies for each of the ks provided as well as the ks used.
    '''
    def knn(self, ks):
        acc = []
        # iterate through each k, and run the training
        for k in ks:
            # setup the model, fit the data, and predit
            model = neighbors.KNeighborsClassifier(n_neighbors=k)
            z = model.fit(self.x_train, self.y_train).predict(self.x_test)
            # add the accuracy to the list
            acc.append(self.get_acc(z, self.y_test))
        return acc, ks

    '''
        Trains and tests the dataset using a mlp neural network using the different number of hidden
        layers provided.
        Returns the list of accuracies and the alphas used.
    '''
    def mlp(self):
        # setup the alphas in logspace
        alphas = numpy.logspace(-10, 3, 10)
        acc = []
        # iterate through the layers and run the training
        for alpha in alphas:
            # setup the model, fit the data, and predit
            model = neural_network.MLPClassifier(solver="lbfgs", activation="tanh", hidden_layer_sizes=(6, 2), 
                                                 alpha=alpha, learning_rate="invscaling", early_stopping=True,
                                                 shuffle=False)
            z = model.fit(self.x_train, self.y_train).predict(self.x_test)
            # add the accuracy to the list
            acc.append(self.get_acc(z, self.y_test))
        return  acc, alphas

    '''
        Trains and tests the dataset using naive bayes with Gaussian, Multinomial, and Bernoulli
        Returns the accuracy for the 3 differents type of methods and the method names.
    '''
    def naive_bayes(self):
        # setup the arrays from the sparse data
        x_train = self.x_train.toarray()
        x_test = self.x_test.toarray()
        # initialize the lists to store the accuracy
        methods = ["Gaussian", "Multinomial", "Bernoulli"]
        acc = []
        # run for gaussian
        gnb = naive_bayes.GaussianNB()
        zg = gnb.fit(x_train, self.y_train).predict(x_test)
        acc.append(self.get_acc(zg, self.y_test))
        # run for multinomial
        mnb = naive_bayes.MultinomialNB()
        zm = mnb.fit(x_train, self.y_train).predict(x_test)
        acc.append(self.get_acc(zm, self.y_test))
        # run for bernoulli
        bnb = naive_bayes.BernoulliNB()
        zb = bnb.fit(x_train, self.y_train).predict(x_test)
        acc.append(self.get_acc(zb, self.y_test))
        return acc, methods

    '''
        Method that normalizes the provided text. Since Indic scripts have a lot of 'quirky behaviour',
        this is needed to canonicalize the representation of text so that the inputs are consistent.
        It also calls the clean_text() function to remove unnecessary data.
    '''
    def normalize_text(self, text):
        text = self.clean_text(text)
        text = self.normalizer.normalize(text)
        return text

    '''
        Returns the component morphemes for the provided text. Although indic_nlp_library does not support
        this for Nepali, the functions are being called for Hindi which should be fairly accurate.
        The text should be cleaned and normalized prior to calling this for the best results.
    '''
    def segment_text(self, text):
        segments = self.morph_analyzer.morph_analyze_document(text.split(' '))
        return segments

    '''
        Trains and tests the dataset using a linear SVM for the provided C values.
        Returns the list of accuracies and the cs used.
    '''
    def svm_lin(self):
        # set the cs using logspace
        cs = numpy.logspace(-10, 3, 10)
        acc = []
        # iterate through the cs and perform different classifications
        for c in cs:
            # setup the svc, fit the training data, and predict
            lin = svm.LinearSVC(C=c)
            z = lin.fit(self.x_train, self.y_train).predict(self.x_test)
            # add the accuracy to the list
            acc.append(self.get_acc(z, self.y_test))
        return acc, cs

    '''
        Trains and tests the dat set using a nuSVC for the provided C values.
        Returns the list of accuracies and the methods(kernels) used.
    '''
    def svm_nu(self):
        # set the methods for the kernel and the acc list
        methods = ['rbf', 'linear', 'poly', 'sigmoid']
        acc = []
        # iterate through the kernel and perform different classifictions
        for method in methods:
            # setup the svc, fit the training data, and predict
            nu = svm.NuSVC(nu=0.1,kernel=method)
            nu.fit(self.x_train, self.y_train)
            z = nu.predict(self.x_test)
            # add the accuracy to the list
            acc.append(self.get_acc(z, self.y_test))
        return acc, methods

    '''
        Trains and tests the dataset using a SVM for the provided C values.
        Returns the list of accuracies and the cs used.
    '''
    def svm_svc(self):
        # set the cs using logspace
        cs = numpy.logspace(-10, 3, 10)
        acc = []
        # iterate through the cs and perform different classifications
        for c in cs:
            # setup the svc, fit the training data, and predict
            svc = svm.SVC(C=c, kernel='rbf')
            z = svc.fit(self.x_train, self.y_train).predict(self.x_test)
            # add the accuracy to the list
            acc.append(self.get_acc(z, self.y_test))
        return acc, cs

    '''
        Method that tokenizes the provided text and returns the list.
        The text should be cleaned and normalized prior to calling this for the best results.
    '''
    def tokenize_text(self, text):
        tokens = self.tokenizer.trivial_tokenize(text)
        return tokens

    '''
        Method that takes a list of messages and creates a transformation matrix out of them.
    '''
    def transform(self, messages):
        dtm = self.vectorizer.transform(messages)
        return dtm