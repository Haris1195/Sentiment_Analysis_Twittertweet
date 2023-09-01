# import random
# import numpy as np
import matplotlib.pyplot as plt

# total_tweets_length = 5
# all_data = []
# for x in range(total_tweets_length):
#     numbers = random.randint(0, 2)
#     all_data.append(numbers)
#     print(x)

    
     
# courses = ['tweet1','tweet2','tweet3','tweet4','tweet5']
# values = all_data
values = [25,0]
values1 = [0,15]

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(values1, values,
        width = 1)

plt.bar(values1, values1, color ='brown',
        width = 1)
        
plt.xlabel("Tweets")
plt.ylabel("Tweet Sentiment Count")
plt.title("Twitter Sentiment Analysis (Total Tweets:40, City: Rawalpindi, Duration: 1hour )")
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # creating the dataset
# data = {'C':20, 'C++':15, 'Java':30,
# 		'Python':35}
# courses = list(data.keys())
# values = list(data.values())

# fig = plt.figure(figsize = (10, 5))

# # creating the bar plot
# plt.bar(courses, values, color ='maroon',
# 		width = 0.4)

# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
# plt.show()
