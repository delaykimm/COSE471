#!/usr/bin/env python
# coding: utf-8

# # Homework 2: Pandas, Regular Expressions, Visualizations
# 
# ## Due Date: Fri 4/14, 11:59 pm KST
# 
# **Collaboration Policy:** You may talk with others about the homework, but we ask that you **write your solutions individually**. If you do discuss the assignments with others, please **include their names** in the following line.
# 
# **Collaborators**: *list collaborators here (if applicable)*

# ### Score Breakdown
# 
# Question | Points
# --- | ---
# Question 1a | 2
# Question 1b | 1
# Question 1c | 2
# Question 2 | 2
# Question 3 | 1
# Question 4 | 2
# Question 5a | 1
# Question 5b | 2
# Question 5c | 2
# Question 6a | 1
# Question 6b | 1
# Question 6c | 1
# Question 6d | 2
# Question 6e | 2
# Total | 22

# ### Initialize your environment
# 
# This cell should run without error.

# In[1]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile
from pprint import pprint # to get a more easily-readable view.

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()
sns.set_context("talk")
import re


# Some common utilities.

# In[2]:


def utils_head(filename, lines=5):
    """
    Returns the first few lines of a file.
    
    filename: the name of the file to open
    lines: the number of lines to include
    
    return: A list of the first few lines from the file.
    """
    from itertools import islice
    with open(filename, "r") as f:
        return list(islice(f, lines))


# # Part 1: Bike Sharing
# 
# The data we are exploring is collected from a bike sharing system in Washington D.C.
# 
# The variables in this data frame are defined as:
# 
# Variable       | Description
# -------------- | ------------------------------------------------------------------
# instant | record index
# dteday | date
# season | 1. spring <br> 2. summer <br> 3. fall <br> 4. winter
# yr | year (0: 2011, 1:2012)
# mnth | month ( 1 to 12)
# hr | hour (0 to 23)
# holiday | whether day is holiday or not
# weekday | day of the week
# workingday | if day is neither weekend nor holiday
# weathersit | 1. clear or partly cloudy <br> 2. mist and clouds <br> 3. light snow or rain <br> 4. heavy rain or snow
# temp | normalized temperature in Celsius (divided by 41)
# atemp | normalized "feels-like" temperature in Celsius (divided by 50)
# hum | normalized percent humidity (divided by 100)
# windspeed| normalized wind speed (divided by 67)
# casual | count of casual users
# registered | count of registered users
# cnt | count of total rental bikes including casual and registered  

# ## Mount your Google Drive
# When you run a code cell, Colab executes it on a temporary cloud instance.  Every time you open the notebook, you will be assigned a different machine.  All compute state and files saved on the previous machine will be lost.  Therefore, you may need to re-download datasets or rerun code after a reset. Here, you can mount your Google drive to the temporary cloud instance's local filesystem using the following code snippet and save files under the specified directory (note that you will have to provide permission every time you run this).

# In[3]:


# mount Google drive
from google.colab import drive
drive.mount('/content/drive')

# now you can see files
get_ipython().system('echo -e "\\nNumber of Google drive files in /content/drive/My Drive/:"')
get_ipython().system('ls -l "/content/drive/My Drive/" | wc -l')
# by the way, you can run any linux command by putting a ! at the start of the line

# by default everything gets executed and saved in /content/
get_ipython().system('echo -e "\\nCurrent directory:"')
get_ipython().system('pwd')


# In[4]:


workspace_path = '/content/drive/MyDrive/COSE471/'  # Change this path!
for line in utils_head(workspace_path+'bikeshare.txt'):
    print(line, end="")


# ### Loading the data
# 
# The following code loads the data into a Pandas DataFrame.

# In[5]:


bike = pd.read_csv(workspace_path+'bikeshare.txt')
bike.head()


# Below, we show the shape of the file. You should see that the size of the DataFrame matches the number of lines in the file, minus the header row.

# In[6]:


bike.shape


# ## Question 1: Data Preparation
# A few of the variables that are numeric/integer actually encode categorical data. These include `holiday`, `weekday`, `workingday`, and `weathersit`. In the following problem, we will convert these four variables to strings specifying the categories. In particular, use 3-letter labels (`Sun`, `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, and `Sat`) for `weekday`. You may simply use `yes`/`no` for `holiday` and `workingday`. 
# 
# In this exercise we will *mutate* the data frame, **overwriting the corresponding variables in the data frame.** However, our notebook will effectively document this in-place data transformation for future readers. Make sure to leave the underlying datafile `bikeshare.txt` unmodified.

# ### Question 1a
# 
# 
# Decode the `holiday`, `weekday`, `workingday`, and `weathersit` fields:
# 
# 1. holiday: Convert to `yes` and `no`. **Hint**: There are fewer holidays...
# 1. weekday: It turns out that Monday is the day with the most holidays.  Mutate the `'weekday'` column to use the 3-letter label (`'Sun'`, `'Mon'`, `'Tue'`, `'Wed'`, `'Thu'`, `'Fri'`, and `'Sat'`) instead of its current numerical values. Note `0` corresponds to `Sun`, `1` to `Mon` and so on.
# 1. workingday: Convert to `yes` and `no`.
# 1. weathersit: You should replace each value with one of `Clear`, `Mist`, `Light`, or `Heavy`.
# 
# **Note:** If you want to revert changes, run the cell that reloads the csv.
# 
# **Hint:**  One simple approach is to use the [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) method of the pandas DataFrame class. We haven't discussed how to do this so you'll need to look at the documentation. The most concise way is with the approach described in the documentation as ``nested-dictonaries``, though there are many possible solutions. E.g. for a DataFrame nested dictionaries, e.g., `{'a': {'b': np.nan}}`, are read as follows: look in column `a` for the value `b` and replace it with `NaN`.
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 2
# -->

# In[7]:


# BEGIN YOUR CODE
# -----------------------
factor_dict = {
    'holiday': {0:"no", 1:"yes"}, 'weekday':{0:"Sun", 1:'Mon', 2:'Tue', 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat"}, 
    'workingday':{0:"no",  1:"yes"}, 'weathersit':{1:"Clear", 2:"Mist", 3:"Light", 4:"Heavy"}
}
# -----------------------
# END YOUR CODE
bike.replace(factor_dict, inplace=True)
bike.head()


# In[8]:


assert isinstance(bike, pd.DataFrame) == True
assert bike['holiday'].dtype == np.dtype('O')
assert list(bike['holiday'].iloc[370:375]) == ['no', 'no', 'yes', 'yes', 'yes']
assert bike['weekday'].dtype == np.dtype('O')
assert bike['workingday'].dtype == np.dtype('O')
assert bike['weathersit'].dtype == np.dtype('O')
assert bike.shape == (17379, 17) or bike.shape == (17379, 18)
assert list(bike['weekday'].iloc[::2000]) == ['Sat', 'Tue', 'Mon', 'Mon', 'Mon', 'Sun', 'Sun', 'Sat', 'Sun']

print('Passed all unit tests!')


# ### Question 1b
# 
# How many entries in the data correspond to holidays?  Set the variable `num_holidays` to this value.
# 
# **Hint:** ``value_counts``
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[9]:


num_holidays = bike['holiday'].value_counts()[1]


# In[10]:


assert num_holidays == 500
assert 1 <= num_holidays <= 10000

print('Passed all unit tests!')


# ### Question 1c (Computing Daily Total Counts)
# 
# The granularity of this data is at the hourly level.  However, for some of the analysis we will also want to compute daily statistics.  In particular, in the next few questions we will be analyzing the daily number of registered and unregistered users.
# 
# Construct a data frame named `daily_counts` indexed by `dteday` with the following columns:
# * `casual`: total number of casual riders for each day
# * `registered`: total number of registered riders for each day
# * `workingday`: whether that day is a working day or not (`yes` or `no`)
# 
# **Hint**: `groupby` and `agg`. For the `agg` method, please check the [documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) for examples on applying different aggregations per column. If you use the capability to do different aggregations by column, you can do this task with a single call to `groupby` and `agg`. For the `workingday` column we can take any of the values since we are grouping by the day, thus the value will be the same within each group. Take a look at the `'first'` or `'last'` aggregation functions.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# points: 2
# -->

# In[11]:


# BEGIN YOUR CODE
# -----------------------
daily_counts = pd.concat([bike.groupby(["dteday"])["casual", "registered"].agg(["sum"]),bike.groupby(["dteday"])["workingday"].agg(["first"])],axis = 1)
daily_counts.columns = ["casual", "registered", "workingday"]
# -----------------------
# END YOUR CODE
daily_counts.head()


# In[12]:


assert np.round(daily_counts['casual'].mean()) == 848.0
assert np.round(daily_counts['casual'].var()) == 471450.0
assert np.round(daily_counts['registered'].mean()) == 3656.0
assert np.round(daily_counts['registered'].var()) == 2434400.0
assert sorted(list(daily_counts['workingday'].value_counts())) == [231, 500]

print('Passed all unit tests!')


# # Part 2: Trump and Tweets
# 
# In this part, we will work with Twitter data in order to analyze Donald Trump's tweets.

# Let's load data into our notebook. Run the cell below to read tweets from the json file into a list named `all_tweets`.

# In[13]:


with open(workspace_path+"hw2-realdonaldtrump_tweets.json", "r") as f:
    all_tweets = json.load(f)


# Here is what a typical tweet from `all_tweets` looks like:

# In[14]:


pprint(all_tweets[-1])


# ## Question 2
# 
# Construct a DataFrame called `trump` containing data from all the tweets stored in `all_tweets`. The index of the DataFrame should be the `ID` of each tweet (looks something like `907698529606541312`). It should have these columns:
# 
# - `time`: The time the tweet was created encoded as a datetime object. (Use `pd.to_datetime` to encode the timestamp.)
# - `source`: The source device of the tweet.
# - `text`: The text of the tweet.
# - `retweet_count`: The retweet count of the tweet. 
# 
# Finally, **the resulting DataFrame should be sorted by the index.**
# 
# **Warning:** *Some tweets will store the text in the `text` field and other will use the `full_text` field.*
# 
# <!--
# BEGIN QUESTION
# name: q1
# points: 2
# -->

# In[15]:


# BEGIN YOUR CODE
# -----------------------
trump = pd.DataFrame(all_tweets).loc[:,["id", "created_at","source", "text", "retweet_count"]] # all_tweets slicing
trump.set_index("id", inplace = True)                         # setting index as id
trump.rename(columns = {'created_at':"time"},inplace = True)  # changing column_name "created_at" to "time"
trump.sort_index(inplace = True)                              # sorted by the index(id)
trump["time"]=pd.to_datetime(trump["time"])                   # change type of trump["time"] to datetime
trump.index.name = None                                       # delete name of index
# -----------------------
# END YOUR CODE
trump.head()


# In[16]:


assert isinstance(trump, pd.DataFrame)
assert 10000 < trump.shape[0] < 11000
assert trump.shape[1] >= 4
assert 831846101179314177 in trump.index
assert all(col in trump.columns for col in ['time', 'source', 'text', 'retweet_count'])
assert np.sometrue([('Twitter for iPhone' in s) for s in trump['source'].unique()])
assert trump['text'].dtype == np.dtype('O')
assert trump['retweet_count'].dtype == np.dtype('int64')
assert 753063644578144260 in trump.index

print('Passed all unit tests!')


# In the following questions, we are going to find out the charateristics of Trump tweets and the devices used for the tweets.
# 
# First let's examine the source field:

# In[17]:


trump['source'].unique()


# ## Question 3
# 
# Notice how sources like "Twitter for Android" or "Instagram" are surrounded by HTML tags. In the cell below, clean up the `source` field by removing the HTML tags from each `source` entry.
# 
# **Hints:** 
# * Use `trump['source'].str.replace` along with a regular expression.
# * You may find it helpful to experiment with regular expressions at [regex101.com](https://regex101.com/).
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 1
# -->

# In[18]:


# BEGIN YOUR CODE
# -----------------------
trump['source'] = trump['source'].str.replace(r"<[^>]+>", '')
# -----------------------
# END YOUR CODE


# In[19]:


assert set(trump['source'].unique()) == set(['Twitter for Android', 'Twitter for iPhone', 'Twitter Web Client',
       'Mobile Web (M5)', 'Instagram', 'Twitter for iPad', 'Media Studio',
       'Periscope', 'Twitter Ads', 'Twitter Media Studio'])

print('Passed all unit tests!')


# In the following plot, we see that there are two device types that are more commonly used than others.

# In[20]:


plt.figure(figsize=(6, 4))
trump['source'].value_counts().plot(kind="bar")
plt.ylabel("Number of Tweets")
plt.title("Number of Tweets by Source");


# ## Question 4
# 
# Now that we have cleaned up the `source` field, let's now look at which device Trump has used over the entire time period of this dataset.
# 
# To examine the distribution of dates we will convert the date to a fractional year that can be plotted as a distribution.
# 
# (Code borrowed from https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years)

# In[21]:


import datetime
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

trump['year'] = trump['time'].apply(year_fraction)


# In[22]:


trump['year'].head()


# Now, use `sns.distplot` to overlay the distributions of Trump's 2 most frequently used web technologies over the years.
# 
# <!--
# BEGIN QUESTION
# name: q3
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[23]:


trump[trump['source']=='Twitter for iPhone']['year']


# In[24]:


# BEGIN YOUR CODE
# -----------------------
top_devices = list(pd.DataFrame(trump['source'].value_counts()).index)[0:2]     # 2 most frequently used web technologies
for device in top_devices:
  data = trump[trump['source']==device]['year']
  sns.distplot(data, label = device.split(" ")[2])

plt.title("Distribution of Tweet Sources Over Years")
plt.legend()
# -----------------------
# END YOUR CODE


# ## Question 5
# 
# 
# Is there a difference between Trump's tweet behavior across these devices? We will attempt to answer this question in our subsequent analysis.
# 
# First, we'll take a look at whether Trump's tweets from an Android device come at different times than his tweets from an iPhone. Note that Twitter gives us his tweets in the [UTC timezone](https://www.wikiwand.com/en/List_of_UTC_time_offsets) (notice the `+0000` in the first few tweets).

# In[25]:


for tweet in all_tweets[:3]:
    print(tweet['created_at'])


# We'll convert the tweet times to US Eastern Time, the timezone of New York and Washington D.C., since those are the places we would expect the most tweet activity from Trump.

# In[26]:


trump['est_time'] = (
    trump['time'].dt.tz_convert("UTC") # Set initial timezone to UTC
                 .dt.tz_convert("EST") # Convert to Eastern Time
)
trump.head()


# ### Question 5a
# 
# Add a column called `hour` to the `trump` table which contains the hour of the day as floating point number computed by:
# 
# $$
# \text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{60^2}
# $$
# 
# * **Hint:** See the cell above for an example of working with [dt accessors](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dt-accessors).
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 1
# -->

# In[27]:


# BEGIN YOUR CODE
# -----------------------
trump['hour'] = trump["est_time"].dt.hour + trump["est_time"].dt.minute/60 + trump["est_time"].dt.second/3600
# -----------------------
# END YOUR CODE


# In[28]:


assert np.isclose(trump.loc[690171032150237184]['hour'], 8.93639) == True

print('Passed all unit tests!')


# ### Question 5b
# 
# Use this data along with the seaborn `distplot` function to examine the distribution over hours of the day in eastern time that trump tweets on each device for the 2 most commonly used devices.
# 
# <!--
# BEGIN QUESTION
# name: q4b
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[29]:


# BEGIN YOUR CODE
# -----------------------
top_devices = list(pd.DataFrame(trump['source'].value_counts()).index)[0:2]
for device in top_devices:
  data = trump[trump['source']==device]['hour']
  sns.kdeplot(data, label = device.split(" ")[2])
  
plt.title("Distribution of Tweet Hours for Different Tweet Sources (pre-2017)")
plt.ylabel("fraction")
plt.legend()
# -----------------------
# END YOUR CODE


# ### Question 5c
# 
# According to [this Verge article](https://www.theverge.com/2017/3/29/15103504/donald-trump-iphone-using-switched-android), Donald Trump switched from an Android to an iPhone sometime in March 2017.
# 
# Let's see if this information significantly changes our plot. Create a figure similar to your figure from question 5b, but this time, only use tweets that were tweeted before 2017.
# 
# <!--
# BEGIN QUESTION
# name: q4c
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[30]:


# BEGIN YOUR CODE
# -----------------------
top_devices =list(pd.DataFrame(trump['source'].value_counts()).index)[0:2]
for device in top_devices:
  data = trump[trump['year']<2017][trump['source']==device]['hour']
  sns.kdeplot(data, label = device.split(" ")[2])

plt.title("Distribution of Tweet Hours for Different Tweet Sources (pre-2017)")
plt.ylabel("fraction")
plt.legend()
# -----------------------
# END YOUR CODE


# ### Question 5d
# 
# During the campaign, it was theorized that Donald Trump's tweets from Android devices were written by him personally, and the tweets from iPhones were from his staff. Does your figure give support to this theory? What kinds of additional analysis could help support or reject this claim?
# 
# <!--
# BEGIN QUESTION
# name: q4d
# points: 1
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# Answer: `My figure rejects this theory.` 
# Distribution of 'Android' using 'looks' same, but the distribution of 'iPhone' using 'seems' to be changed after Trump switched his phone from an Android to an iPhone in 2017. Trump's changing phone(stop using 'Android') doesn't affect to the distribution of 'Android' using. That means even before 2017, Trump didn't use 'Android'. Therefore, it rejects the theory that Donald Trump's tweets from Android devices were written by him personally. It seems like the tweets from Android were from his staff, not Trump. To support my idea, I should check whether there is significant difference between the distribution of 'Android' with entire data and that of 'Android' with data only before 2017. It can be checked by 'Significant tests'.

# ---
# # Part 3: Sentiment Analysis
# 
# It turns out that we can use the words in Trump's tweets to calculate a measure of the sentiment of the tweet. For example, the sentence "I love America!" has positive sentiment, whereas the sentence "I hate taxes!" has a negative sentiment. In addition, some words have stronger positive / negative sentiment than others: "I love America." is more positive than "I like America."
# 
# We will use the [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment) lexicon to analyze the sentiment of Trump's tweets. VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media which is great for our usage.
# 
# The VADER lexicon gives the sentiment of individual words. Run the following cell to show the first few rows of the lexicon:

# In[31]:


print(''.join(open(workspace_path+"vader_lexicon.txt").readlines()[:10]))


# ## Question 6
# 
# As you can see, the lexicon contains emojis too! Each row contains a word and the *polarity* of that word, measuring how positive or negative the word is.
# 
# (How did they decide the polarities of these words? What are the other two columns in the lexicon? See the link above.)
# 
# ### Question 6a
# 
# Read in the lexicon into a DataFrame called `sent`. The index of the DataFrame should be the words in the lexicon. `sent` should have one column named `polarity`, storing the polarity of each word.
# 
# * **Hint:** The `pd.read_csv` function may help here. 
# 
# <!--
# BEGIN QUESTION
# name: q5a
# points: 1
# -->

# In[32]:


# BEGIN YOUR CODE
# -----------------------
sent = pd.DataFrame(pd.read_csv(workspace_path+'vader_lexicon.txt', sep = '\t', header = None, index_col = 0, names= ["polarity", 2,3])["polarity"])
# -----------------------
# END YOUR CODE
sent.head()


# In[33]:


assert np.allclose(sent['polarity'].head(), [-1.5, -0.4, -1.5, -0.4, -0.7]) == True
assert list(sent.index[5000:5005]) == ['paranoids', 'pardon', 'pardoned', 'pardoning', 'pardons']

print('Passed all unit tests!')


# ### Question 6b
# 
# Now, let's use this lexicon to calculate the overall sentiment for each of Trump's tweets. Here's the basic idea:
# 
# 1. For each tweet, find the sentiment of each word.
# 2. Calculate the sentiment of each tweet by taking the sum of the sentiments of its words.
# 
# First, let's lowercase the text in the tweets since the lexicon is also lowercase. Set the `text` column of the `trump` DataFrame to be the lowercased text of each tweet.
# 
# <!--
# BEGIN QUESTION
# name: q5b
# points: 1
# -->

# In[34]:


# BEGIN YOUR CODE
# -----------------------
trump['text']= trump['text'].str.lower()
# -----------------------
# END YOUR CODE
trump.head()


# In[35]:


assert trump['text'].loc[884740553040175104] == 'working hard to get the olympics for the united states (l.a.). stay tuned!'

print('Passed all unit tests!')


# ### Question 6c
# 
# Now, let's get rid of punctuation since it will cause us to fail to match words. Create a new column called `no_punc` in the `trump` DataFrame to be the lowercased text of each tweet with all punctuation replaced by a single space. We consider punctuation characters to be **any character that isn't a Unicode word character or a whitespace character**. You may want to consult the Python documentation on regexes for this problem.
# 
# (Why don't we simply remove punctuation instead of replacing with a space? See if you can figure this out by looking at the tweet data.)
# 
# <!--
# BEGIN QUESTION
# name: q5c
# points: 1
# -->

# In[36]:


# BEGIN YOUR CODE
# -----------------------
punct_re = r'[^0-~]'  # Save your regex in punct_re
trump['no_punc'] = trump['text'].str.replace(punct_re, " ")
trump['no_punc']
# -----------------------
# END YOUR CODE


# In[37]:


assert re.search(punct_re, 'this') == None
assert re.search(punct_re, 'this is not ok.') != None
assert re.search(punct_re, 'this#is#ok') != None
assert re.search(punct_re, 'this^is ok') != None

print('Passed all unit tests!')


# ### Question 6d
# 
# Now, let's convert the tweets into what's called a [*tidy format*](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) to make the sentiments easier to calculate. Use the `no_punc` column of `trump` to create a table called `tidy_format`. The index of the table should be the IDs of the tweets, repeated once for every word in the tweet. It has two columns:
# 
# 1. `num`: The location of the word in the tweet. For example, if the tweet was "i love america", then the location of the word "i" is 0, "love" is 1, and "america" is 2.
# 2. `word`: The individual words of each tweet.
# 
# The first few rows of our `tidy_format` table look like:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>num</th>
#       <th>word</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>894661651760377856</th>
#       <td>0</td>
#       <td>i</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>1</td>
#       <td>think</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>2</td>
#       <td>senator</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>3</td>
#       <td>blumenthal</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>4</td>
#       <td>should</td>
#     </tr>
#   </tbody>
# </table>
# 
# **Note that your DataFrame may look different from the one above.** However, you can double check that your tweet with ID `894661651760377856` has the same rows as ours. Our tests don't check whether your table looks exactly like ours.
# 
# As usual, try to avoid using any for loops. Our solution uses a chain of 5 methods on the `trump` DataFrame, albeit using some rather advanced Pandas hacking.
# 
# * **Hint 1:** Try looking at the `expand` argument to pandas' `str.split`.
# 
# * **Hint 2:** Try looking at the `stack()` method.
# 
# * **Hint 3:** Try looking at the `level` parameter of the `reset_index` method.
# 
# <!--
# BEGIN QUESTION
# name: q5d
# points: 2
# -->

# In[38]:


# BEGIN YOUR CODE
# -----------------------
tidy_format = pd.DataFrame(trump["no_punc"].str.split(expand = True).stack()).reset_index(level = 1)
tidy_format.columns = ["num", "word"]
# -----------------------
# END YOUR CODE
tidy_format.head()


# In[39]:


assert tidy_format.loc[894661651760377856].shape == (27,2)
assert ' '.join(list(tidy_format.loc[894661651760377856]['word'])) == 'i think senator blumenthal should take a nice long vacation in vietnam where he lied about his service so he can at least say he was there'

print('Passed all unit tests!')


# ### Question 6e
# 
# Now that we have this table in the tidy format, it becomes much easier to find the sentiment of each tweet: we can join the table with the lexicon table. 
# 
# Add a `polarity` column to the `trump` table.  The `polarity` column should contain the sum of the sentiment polarity of each word in the text of the tweet.
# 
# **Hints:** 
# * You will need to merge the `tidy_format` and `sent` tables and group the final answer.
# * If certain words are not found in the `sent` table, set their polarities to 0.
# 
# <!--
# BEGIN QUESTION
# name: q5e
# points: 2
# -->

# In[40]:


# BEGIN YOUR CODE
# -----------------------
tidy_format["word_polarity"]=[sent.loc[word, "polarity"] if word in sent.index else 0 for word in tidy_format["word"]]
trump['polarity'] = tidy_format["word_polarity"].reset_index().groupby("index").sum()
trump["polarity"]=pd.DataFrame(trump["polarity"]).apply(pd.to_numeric, errors = "coerce").fillna(0)    # object -> float64로 형변환환
trump.dropna(inplace = True)
# -----------------------
# END YOUR CODE
trump[['text', 'polarity']].head()


# In[41]:


assert np.allclose(trump.loc[744701872456536064, 'polarity'], 8.4)
assert np.allclose(trump.loc[745304731346702336, 'polarity'], 2.5)
assert np.allclose(trump.loc[744519497764184064, 'polarity'], 1.7)
assert np.allclose(trump.loc[894661651760377856, 'polarity'], 0.2)
assert np.allclose(trump.loc[894620077634592769, 'polarity'], 5.4)

print('Passed all unit tests!')


# Now we have a measure of the sentiment of each of his tweets! Note that this calculation is rather basic; you can read over the VADER readme to understand a more robust sentiment analysis.
# 
# Now, run the cells below to see the most positive and most negative tweets from Trump in your dataset:

# In[42]:


print('Most negative tweets:')
for t in trump.sort_values('polarity').head()['text']:
    print('\n  ', t)


# In[43]:


print('Most positive tweets:')
for t in trump.sort_values('polarity', ascending=False).head()['text']:
    print('\n  ', t)


# ---
# 
# Now, let's try looking at the distributions of sentiments for tweets containing certain keywords.
# 
# In the cell below, we create a single plot showing both the distribution of tweet sentiments for tweets containing `nytimes`, as well as the distribution of tweet sentiments for tweets containing `fox`. Here, we notice that the president appears to say more positive things about Fox than the New York Times.

# In[44]:


sns.distplot(trump[trump['text'].str.lower().str.contains("nytimes")]['polarity'], label = 'nytimes')
sns.distplot(trump[trump['text'].str.lower().str.contains("fox")]['polarity'], label = 'fox')
plt.title('Distributions of Tweet Polarities (nytimes vs. fox)')
plt.legend();


# ### Congratulations! You have completed HW2.
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.,
# 
# Please generate pdf as follows and submit it to Gradescope.
# 
# **File > Print Preview > Print > Save as pdf**
# 
# **Please save before submitting!**
# 
# <!-- EXPECT 5 EXPORTED QUESTIONS -->

# In[44]:




