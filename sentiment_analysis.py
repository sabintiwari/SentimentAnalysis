# import the third party libraries
import initialize
initialize.setup_paths()

# Class that handles the sentiment analysis.
class SentimentAnalysis(object):

    # Method that cleans the message using regex.
    def clean_text(self, text):
        a = 0