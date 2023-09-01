import matplotlib.pyplot as plt
import numpy as np
data = {'Rawalpindi1__pos1':35, 'Rawalpindi_neg1':15, 'Lahore_pos2':1, 'lahore__neg2':34}
# data = all_tweets
all_dict  = []
for x in data.keys():
    all_dict.append(x)
X_dict = all_dict[0::2]

updated_dict = []
for city in X_dict:
    if 'Islamabad' in city:
        updated_dict.append('PPP')
        print(updated_dict)

    if 'Rawalpindi' in city:
        updated_dict.append('PMLN')
        print(updated_dict)

    if 'Karachi' in city:
        updated_dict.append('PTI')
        print(updated_dict)
    if 'Lahore' in city:
        updated_dict.append('PTI')
        print(updated_dict)
    if 'Faisalabad' in city:
        updated_dict.append('PPP')
        print(updated_dict)
    if 'Peshawar' in city:
        updated_dict.append('PMLN')
        print(updated_dict)
    
    print('city',city)


neg_tweet = []
pos_tweet = []
for x, y in data.items():
    if 'pos' in x:
        print('pos aval')
        pos_tweet.append(y)
    else:
        neg_tweet.append(y)
    print(x, y)
print(pos_tweet)
print(neg_tweet)
remove_neg_tweets = []
for check_neg in neg_tweet:
    if check_neg < 0:
        x = check_neg * -1
        remove_neg_tweets.append(x)
    else:
        remove_neg_tweets.append(check_neg)
        
        print('neg save vlaue')
    print('remve neg')

remove_pos_tweets = []
for check_pos in pos_tweet:
    if check_pos < 0:
        x = check_pos * -1
        remove_pos_tweets.append(x)
    else:
        remove_pos_tweets.append(check_pos)
        
        print('neg save vlaue')
    print('remve neg')
print('remove negative sign value', remove_pos_tweets)
print('remove negative sign value', remove_neg_tweets)
    
X = updated_dict

negative_tweets_number = remove_neg_tweets
positive_tweets_number = remove_pos_tweets

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, positive_tweets_number, 0.4, label = 'Positive')
plt.bar(X_axis + 0.2, negative_tweets_number, 0.4, label = 'Negative')
ax = plt.gca()
y_label = "0    0.1  0.2    0.3   0.4   0.5   0.6   0.7   0.8   0.9   1"
plt.ylabel(y_label)   
ax.tick_params(axis="y", colors="white")
plt.xticks(X_axis, X)
plt.legend()
plt.savefig('sentimentapp\static\Compersion_graph.jpg')
print("Compersion_graph saved successfully")
plt.show()    