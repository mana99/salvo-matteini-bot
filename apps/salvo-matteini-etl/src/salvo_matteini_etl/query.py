import abc

from salvo_matteini_etl.authenticator import access_token


class Request(metaclass=abc.ABCMeta):

    @property
    def method(self):
        raise NotImplementedError("Missing method property")    # todo available methods

    @property
    def url(self):
        raise NotImplementedError("Missing url property")

    @property
    def headers(self):
        raise NotImplementedError("Missing headers property")

    @property
    def params(self):
        raise NotImplementedError("Missing params property")


class SearchTweets(Request):

    method = "GET"
    url = "https://api.twitter.com/1.1/search/tweets.json"
    headers = {"Authorization": f"Bearer {access_token}"}

    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        return self._params
