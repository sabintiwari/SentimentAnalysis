# import system libraries
import datetime
import io
import json
import os
import sys
import time

# import the third party libraries
import initialize
initialize.setup_paths()
import tweepy

# Main Method
def main(args):
    # read the configs from the file
    config = get_config()
    count = config["data_count"]
    max_id = config["max_tweet_id"]
    filename = config["filename"]

    # clear the log file for FetchData
    io.open("logs\\FetchData.py.log", "w").close()

    # setup the twitter authorization
    auth = tweepy.OAuthHandler(config["consumer_key"], config["consumer_secret"])
    auth.set_access_token(config["access_token"], config["access_token_secret"])

    # get the API
    log("Connecting to the twitter API...")
    api = tweepy.API(auth, wait_on_rate_limit=True)
    log("Success!\n")

    # get the user from the api
    log("Getting the user object with screen name: " + config["user"])
    user = api.get_user(config["user"])
    total = 0
    log("Success!\n")
    try:
        # try to get the timeline for the user
        check_rate_limit(api)
        tweet_ids = []
        if max_id != 0:
            # if the max id is provided, get tweets that have the max id to the value
            log("Getting " + str(count) + " statuses with ids that are less than " + str(max_id))
            # get the replies for each status received
            for status in tweepy.Cursor(api.user_timeline, id=user.id, count=count, max_id=max_id).items():
                tweet_ids.append(status.id)
        else:
            # else get the most recent tweets
            log("Getting " + str(count) + " most recent statuses")
            # get the replies for each status received
            for status in tweepy.Cursor(api.user_timeline, count=count, id=user.id).items():
                tweet_ids.append(status.id)
        # get all the replies since the lowest id
        min_id = min(tweet_ids)
        replies = get_all_replies(api, min_id, user.screen_name)
        total = add_tweets_to_file(replies, filename, total)
    except tweepy.TweepError:
        # if there is a ratelimit error, sleep for 15 minutes and try again
        log("Twitter API rate limit has been reached")

    # log the total number of data retrieved
    log(str(total) + " total number of replies added since tweet id " + str(min_id))


# Helper Functions

# Adds the provided tweet replies to the file that was passed.
def add_tweets_to_file(replies, filepath, total):
    file = io.open(filepath, "a+", encoding="utf8")
    for reply in replies:
        total = total + 1
        file.write("\"" + reply.full_text + "\"\n")
    file.close()
    return total

# Checks the rate limit and current remaining requests. Waits if the limit has been reached and
#  the True flag is passed for the 'wait' parameter.
def check_rate_limit(api, wait=True, buffer=0.1):
    log("Checking if the rate limit for the API has been reached.")
    # get the number of remaining requests
    rem = int(api.last_response.headers["x-rate-limit-remaining"])
    if rem == 0:
        # get the limit and reset values from the headers
        limit = int(api.last_response.headers["x-rate-limit-limit"])
        reset = int(api.last_response.headers["x-rate-limit-reset"])
        # get the time and print the time requests remaining for the current reset
        reset = datetime.datetime.fromtimestamp(reset)
        log("0 out of {} remaining. Reset at {}".format(limit, reset))
        if wait:
            # wait till the limit is reset until if the wait flag is passed
            delay = (reset - datetime.datetime.now()).total_seconds() + buffer
            log("Waiting for {} second(s)".format(delay))
            time.sleep(delay)
            return True
        else:
            # return false when requests cannot be made
            return False

# Gets all the replies to since the provided minimum tweet id.
def get_all_replies(api, min_id, user):
    # setup the query using the tweet id and username
    log("Getting all replies since  the tweet with id: " + str(min_id))
    query = "to:" + user
    replies = []
    # wait for rate limit to replenish and call the search query
    check_rate_limit(api)
    # for each result check if the reply matches the current tweet
    for result in tweepy.Cursor(api.search, q=query, since_id=str(min_id), tweet_mode="extended", count=1000).items():
        replies.append(result)
    #return the list of replies
    return replies

# Gets the config json object from the provided file
def get_config():
    file = io.open("config.json")
    text = file.read()
    file.close()
    return json.loads(text)

# Logs the provided message to the 
def log(message):
    print(message + "\n")
    file = io.open("logs\\FetchData.py.log", "a+")
    file.write(message + "\n")
    file.close()

# Call the main method
main(sys.argv)