
import requests
from requests_oauthlib import OAuth1
from os import environ


TWITTER_ACCESS_TOKEN = environ["TWITTER_ACCESS_TOKEN"]
TWITTER_ACCESS_TOKEN_SECRET = environ["TWITTER_ACCESS_TOKEN_SECRET"]
TWITTER_CONSUMER_KEY = environ["TWITTER_CONSUMER_KEY"]
TWITTER_CONSUMER_KEY_SECRET = environ["TWITTER_CONSUMER_KEY_SECRET"]

BASE_URL = "https://api.twitter.com/"
ENDPOINT = "1.1/search/tweets.json"


url = BASE_URL + ENDPOINT
auth = OAuth1(client_key=TWITTER_CONSUMER_KEY,
              client_secret=TWITTER_CONSUMER_KEY_SECRET,
              resource_owner_key=TWITTER_ACCESS_TOKEN,
              resource_owner_secret=TWITTER_ACCESS_TOKEN_SECRET,
              signature_method="HMAC-SHA1")
query_params = {
    "q": "salvini -filter:media",
    "result_type": "recent",
    "lang": "it",
    "tweet_mode": "extended",
    "count": 5,
}

resp = requests.get(url, auth=auth, params=query_params)
j = resp.json()

print(j)
