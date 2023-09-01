import matplotlib.pyplot as plt
import random

Total_tweets = 40
city = "Lahore"
Duration = "1-hour"

numbers = random.randint(2, 4)
negtive = Total_tweets/numbers
positive = Total_tweets - negtive

# graph_title = "Twitter Sentiment Analysis (Total Tweets:%s, City: %s, Duration: %s )"%(Total_tweets,city,Duration)
graph_title = "Twitter Sentiment Analysis"

values = [1,0]
values1 = [0,0]

if values1[1]==0:
        plt.xlabel("Positive")
else:
    plt.xlabel("Positive                                             Negative")
fig = plt.figure()

# creating the bar plot
plt.bar(values1, values,
        width = .2)

plt.bar(values1, values1, color ='brown',
        width = .2)
        
# plt.xlabel("Tweets")
# plt.xlabel("Positive                                             Negative")
plt.ylabel("Tweet Sentiment Polarity")
ax = plt.gca()
ax.tick_params(axis="x", colors="white")
plt.title(graph_title)
plt.show()