import numpy as np
import matplotlib.pyplot as plt
import random


total_tweets_length = 5
N = total_tweets_length
all_data = []

for x in range(total_tweets_length):
    numbers = random.choice([0,25])
    # numbers = random.randint(0, 25)
    all_data.append(numbers)
    print(x)

values1 = all_data

values2 = []
for value in values1:
    if value>1:
        values2.append(0)
    else:
        values2.append(25)

ind = np.arange(N) # the x locations for the groups
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind, values1, width, color='r')
ax.bar(ind, values2, width,bottom=values1, color='b')
ax.set_ylabel('Sentiment intent')
ax.set_title('Sentiment')
# ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
ax.set_yticks(np.arange(0, 81, 10))
ax.legend(labels=['Negtive', 'Postive'])
plt.show()