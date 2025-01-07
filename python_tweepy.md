Python Tweepy — a complete guide
Skillcate AI
Skillcate AI

·
Follow

5 min read
·
Aug 22, 2022
6


1





Making the most of your Twitter API

In this tutorial, I will give you a quick walkthrough to Tweepy: which is an easy-to-use Python library for accessing the Twitter API.

Once you have configured your Twitter API credentials, next thing you want is to build a Bot or to pull tweets for sentiment analysis, etc. In this course, I’ll give you a complete guide to use Tweepy for performing read & write operations, like: posting tweets, deleting tweets, following someone, unfollowing someone, liking a tweet, retweeting, etc.

Watch the video tutorial instead
If you are more of a video person, go ahead and watch it on YouTube, instead. FYI, we launch new machine learning projects every week. So, make sure to subscribe to our channel to get access to all of our free ML courses.


All project related files are kept on Google Drive. On this note, let’s get started..

Getting started with Tweepy
To start off, you may require Twitter API credentials, which you may have by now. Twitter has couple of API versions: V1.1 and V2.0. Following the Tweepy documentation, we shall do the following:

a. Twitter API v1.1
Authenticate using OAuth 1.0a — User Context
Using Tweepy Class API
To perform write operations on the Twitter A/c via API, like: posting tweets, deleting tweets, following someone, unfollowing someone, liking a tweet, retweeting, etc.
b. Twitter API v2.0
Authenticate using OAuth 2.0 — Bearer Token (App only)
Using Tweepy Class Client
To pull tweets on a specific keyword to perform text analytics

For your ready reference, I’ve kept all project related files here in this Google Drive folder. Over here, b1_tweepy_walkthrough is our Jupyter Notebook for this tutorial.

Our high-level plan-of-action is to:

Pull ~1000 tweets from Twitter on a certain trending keyword, like: #apple,
Perform sentiment analysis on Tweets, using: Python TextBlob,
Create beautiful visualisation to draw inferences

Tweepy Code Walkthrough
To get started with Tweepy, as a first step, we need to set up our environment. Here, we are basically installing & then importing tweepy
!pip install tweepy==4.9.0
import tweepy
Performing write operations
Next up, we are first authenticating for Twitter API v1.1. So, we authenticate using OAuth 1.0a — User Context. Update your API creds below as required
# Authenticate with Twitter OAuth 1.0a User Context
auth = tweepy.OAuth1UserHandler(
# API / Consumer Key here
"replace_me",
# API / Consumer Secret here
"replace_me",
# Access Token here
"replace_me",
# Access Token Secret here
"replace_me"
)
api = tweepy.API(auth)
Let’s start off with making a tweet telling, “Hello world!”
tweet = api.update_status("Hello world!")
Now, let’s pull some tweets from your home timeline. Home timeline is your Twitter home basically, that has tweets from the people you follow & your own recent tweets
# Get recent tweets from your home timeline
tweets = api.home_timeline(count=5)
for tweet in tweets:
print(tweet.text,'\n')
Next up, let’s get some of the recent tweets posted by you on your account, using the user_timeline method
# Get Tweets (recent tweets from your account)
tweets = api.user_timeline(count=2)
for tweet in tweets:
print(tweet.text,'\n')
When we pull tweets from twitter, there is additional information that is passed along with the tweet, like the creation date, tweet id, etc. Let’s try printing some of this information here..
# Tweet object has lots of data, let's check Tweet JSON specs
tweets = api.user_timeline(count=2)
for tweet in tweets:
print(tweet.created_at)
print(tweet.id_str)
print(tweet.text,'\n')
Tweet ID (id_str) is a unique identifier for every tweet posted on the platform. Now, using this Tweet ID, let’s try deleting a tweet. For this, first up, I create a tweet
# Create a tweet to delete later
to_be_deleted_tweet = api.update_status('Bad Tweet')
Now, using the tweet ID, let’s perform the delete operation
# Delete the Bad Tweet
api.destroy_status(to_be_deleted_tweet.id_str)
You may check on your Twitter account if changes are made.

You may follow an account with create_friendship method. For this, I am using YouTube’s Twitter handle, that I’m not following for now..
# Follow
api.create_friendship(screen_name='@YouTube')
Similar way, you may unfollow as well..
# Unfollow
api.destroy_friendship(screen_name='@YouTube')
Alright, so this was Twitter API v1.1, with which we performed write operations.

Pulling tweets (read operation)
Now let’s jump onto the Twitter API v2.0, and try pulling some tweets

For this, the first step is authentication. We are using OAuth 2.0 this time, that grants us Read-Only access. Which is fine absolutely fine for pulling public tweets
# Authenticate with OAuth 2.0 Bearer Token (App only)
client = tweepy.Client(bearer_token='replace_me')
Next up, we are pulling recent tweets on the hashtag #elonmusk, with these specific directions, meaning: I don’t want retweets, I want tweets in english language, and I want maximum of 10 results
# Pull tweets from twitter
query = '#elonmusk -is:retweet lang:en'
tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
# Get tweets that contain the hashtag #TypeKeywordHere
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
# print pulled tweets
for tweet in tweets.data:
print('\n**Tweet Text**\n',tweet.text)
Well, these are only some of the many operations Tweepy may perform for us. To try out more Tweepy operations, like: retweeting, etc., you may go through the Tweepy documentation. In case you get stuck somewhere, or have doubts, post them in the comments section below, and I’ll help you out.

Brief about Skillcate
At Skillcate, we are on a mission to bring you application based machine learning education. We launch new machine learning projects every week. So, make sure to subscribe to our YouTube channel and also hit that bell icon, so you get notified when our new ML Projects go live.

Shall be back soon with a new ML project. Until then, happy learning 🤗!!