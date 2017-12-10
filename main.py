# import libraries and initialize them
import initialize
initialize.setup_paths()
import io
import sentiment_analyzer
import sys

'''
    The entry method for the application that runs the analyzer.
'''
def main():
    sa = sentiment_analyzer.SentimentAnalyzer()

    input_text = u"@bbcnepali @brb1954 जनता सधै खुसिक कुरा चहान पुरा  गरि दिनु"
    print("Original Text: " + input_text)
    input_text = sa.normalize_text(input_text)
    print("Normalized Text: " + input_text)

# Call the main method
main()