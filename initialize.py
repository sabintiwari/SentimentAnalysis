import os
import sys

'''
    This script has all the setup steps that are used to initialize everything that is needed to run the code
'''

'''
    Sets up the paths and resources that are needed for the third party libraries.
'''
def setup_paths():
    # get the working directory
    work_dir = os.getcwd()
    
    # add the path for tweepy
    sys.path.insert(0, "{}\\libraries\\tweepy.egg".format(work_dir))

    # add the path for indic_nlp_library and setup the resources
    sys.path.insert(0, "{}\\libraries\\indic_nlp_library.egg".format(work_dir))
    from indicnlp import common
    common.set_resources_path("{}\\libraries\\indic_nlp_resources".format(work_dir))

    # add the path for numpy and scipy
    sys.path.insert(0, "{}\\libraries\\numpy.egg".format(work_dir))
    sys.path.insert(0, "{}\\libraries\\scipy.egg".format(work_dir))