
# https://benalexkeen.com/interacting-with-the-twitter-api-using-python/


import base64
import requests
from os import environ
#from requests_oauthlib import OAuth1Session

BASE_URL = 'https://api.twitter.com/'
CLIENT_KEY = environ["TWITTER_CONSUMER_KEY"]
CLIENT_SECRET = environ["TWITTER_CONSUMER_KEY_SECRET"]


def get_bearer_token(client_key, client_secret):
    """
    POST oauth2/token
    https://developer.twitter.com/en/docs/authentication/api-reference/token

    :param client_key:
    :param client_secret:
    :return:
    """
    auth_url = BASE_URL + "oauth2/token"

    key_secret = f"{client_key}:{client_secret}".encode("ascii")
    b64_encoded_key = base64.b64encode(key_secret)
    b64_encoded_key = b64_encoded_key.decode("ascii")
    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }
    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)
    return auth_resp


def revoke_bearer_token(access_token, client_key, client_secret):  #fixme
    """
     POST oauth2/invalidate_token
     https://developer.twitter.com/en/docs/authentication/api-reference/invalidate_bearer_token

    :return:
    """

    auth_url = BASE_URL + "oauth2/invalidate_token"
    auth_params = {
        "access_token": access_token
    }
    oauth = f'oauth_consumer_key="{client_key}", ' \
            f'oauth_nonce="AUTO_GENERATED_NONCE", ' \
            f'oauth_signature="AUTO_GENERATED_SIGNATURE", ' \
            f'oauth_signature_method="HMAC-SHA1", ' \
            f'oauth_timestamp="AUTO_GENERATED_TIMESTAMP", ' \
            f'oauth_token="{access_token}", ' \
            f'oauth_version="1.0"'
    auth_headers = {'Authorization': f'OAuth {oauth}'}

    auth_resp = requests.post(auth_url, headers=auth_headers, params=auth_params)

    return auth_resp


resp = get_bearer_token(CLIENT_KEY, CLIENT_SECRET)
access_token = resp.json()['access_token']
# resp = revoke_bearer_token(access_token, CLIENT_KEY, CLIENT_SECRET)
# print(resp.status_code)
# print(resp.content)

