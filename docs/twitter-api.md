
## Search

Standard Search: This search API searches against a sampling of recent Tweets published 
in the past 7 days. Part of the 'public' set of APIs. [^1]


Resource URL: https://api.twitter.com/1.1/search/tweets.json

Auth method: OAuth 1.0

Resource Information 

* Response formats	JSON
* Requires authentication?	Yes
* Rate limited?	Yes
* Requests / 15-min window (user auth)	180
* Requests / 15-min window (app auth)	450

## Parameters

https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets


### Query parameter

https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/guides/standard-operators

| Operator | Finds Tweets... |
| --- | --- |
| puppy filter:media | containing “puppy” and an image or video. |
| puppy \-filter:retweets | containing “puppy”, filtering out retweets |
| puppy filter:native\_video | containing “puppy” and an uploaded video, Amplify video, Periscope, or Vine. |
| puppy filter:periscope | containing “puppy” and a Periscope video URL. |
| puppy filter:vine | containing “puppy” and a Vine. |
| puppy filter:images | containing “puppy” and links identified as photos, including third parties such as Instagram. |
| puppy filter:twimg | containing “puppy” and a pic.twitter.com link representing one or more photos. |
| superhero since:2015\-12\-21 | containing “superhero” and sent since date “2015\-12\-21” (year\-month\-day). |
| puppy until:2015\-12\-21 | containing “puppy” and sent before the date “2015\-12\-21”. |
| movie \-scary :) | containing “movie”, but not “scary”, and with a positive attitude. |
| flight :( | containing “flight” and with a negative attitude. |
| traffic ? | containing “traffic” and asking a question. |


[^1]: https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/overview



