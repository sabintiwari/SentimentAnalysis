# import default libraries
import re
import sys

# import the indic nlp library
from indicnlp import common
from indicnlp import loader
from indicnlp.morph import unsupervised_morph
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# import the sklearn library
from sklearn.feature_extraction.text import CountVectorizer

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
        Initialization function
    '''
    def __init__(self):
        # initialize the indic nlp library components
        loader.load()
        self.factory = IndicNormalizerFactory()
        # self.morph_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('hi')
        self.normalizer = self.factory.get_normalizer("ne", False)
        self.tokenizer = indic_tokenize

        # initialize the sklearn library components
        self.vectorizer = CountVectorizer(binary=False, encoding=u'utf-8', input=u'content', lowercase=False, tokenizer=self.tokenize_text)

    '''
        Method that cleans the message by removing unncessary information using regex.
        This removes HTML tags, mentions, hashtags, links, and leading/trailing whitespaces.
    '''
    def clean_text(self, text):
        for token in self.ignore_tokens:
            text = re.sub(token, "", text)
        return text.strip()

    '''
        Calls the fit() method of the CountVectorizer on the collection of text values.
        Return the features that are learned from the CountVectorizer.
    '''
    def fit(self, text_collection):
        # normalize the text first
        messages = []
        for item in text_collection:
            text = self.normalize_text(item)
            messages.append(text)
        # call the fit method of the vectorizer
        self.vectorizer.fit(messages)
        return self.vectorizer.get_feature_names()

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
        Method that tokenizes the provided text and returns the list.
        The text should be cleaned and normalized prior to calling this for the best results.
    '''
    def tokenize_text(self, text):
        tokens = self.tokenizer.trivial_tokenize(text)
        return tokens