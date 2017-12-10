# import default libraries
import re
import sys

# import the third party libraries
from indicnlp import loader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

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

    ''' 
        Initialization function
    '''
    def __init__(self):
        # initialize the indic nlp library components
        loader.load()
        self.factory = IndicNormalizerFactory()
        self.normalizer = self.factory.get_normalizer("ne", False)

    '''
        Method that cleans the message by removing unncessary information using regex.
        This removes HTML tags, mentions, hashtags, links, and leading/trailing whitespaces.
    '''
    def clean_text(self, text):
        for token in self.ignore_tokens:
            text = re.sub(token, "", text)
        return text.strip()

    '''
        Method that normalizes the provided text. Since Indic scripts have a lot of 'quirky behaviour',
        this is needed to canonicalize the representation of text so that the inputs are consistent.
        It also calls the clean_text() function to remove unnecessary data.
    '''
    def normalize_text(self, text):
        text = self.clean_text(text)
        text = self.normalizer.normalize(text)
        return text