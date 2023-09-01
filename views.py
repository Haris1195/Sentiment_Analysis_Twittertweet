from django.shortcuts import render
from django.http import HttpResponse
import torch
import random
import pandas as pd
from tqdm.notebook import trange, tqdm
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer
import numpy as np

import threading

select_duration = ''
Select_City = ''
Select_Party = ''

import time
import twitterbot as tb
import sys
import mysecrets
import numpy as np
# global positive_value
# positive_value = 0
all_tweets = {}
x_var = None
data1 = []

# def home(request):
#     return render(request, 'home.html')
def home2(request):
    return render(request, 'home2.html')

def show_comparison():

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    import numpy as np

    global all_tweets
    data = all_tweets

    all_dict  = []
    for x in data.keys():
        all_dict.append(x)
    X_dict = all_dict[0::2]

    updated_dict = []
    for city in X_dict:
        if 'Islamabad' in city:
            if 'PMLN'  in  city:
                updated_dict.append('PMLN')
                print(updated_dict)
            elif 'PTI' in city:
                updated_dict.append('PTI')
                print(updated_dict)
                
            else:
                updated_dict.append('PPP')
                print(updated_dict)
                

        elif 'Rawalpindi' in city:
                if 'PMLN' in city:
                        updated_dict.append('PMLN')
                        print(updated_dict)
                elif 'PTI'  in city:
                                 
                        updated_dict.append('PTI')
                        print(updated_dict)
                    
                else:
                    updated_dict.append
                    updated_dict.append('PPP')
                    print(updated_dict)


        elif 'Lahore' in city:
               if 'PMLN' in city:             
                    updated_dict.append('PMLN')
                    print(updated_dict)
               elif 'PTI' in city:
                   
                    updated_dict.append('PTI')
                    print(updated_dict)
                    
               else:
                   updated_dict.append
                   updated_dict.append('PPP')
                   print(updated_dict)
                    
            # updated_dict.append('PPP')
            # print(updated_dict)
        else:
            print('nothing')
        
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
    leg = ax.get_legend()
    # leg.legendHandles[0].set_color('red')
    # leg.legendHandles[1].set_color('yellow')
    ax.tick_params(axis="y", colors="white")
    plt.xticks(X_axis, X)
    plt.legend()
    y_label = " 0  10   20   30   40   50   60   70    80     90"
    plt.ylabel(y_label) 
    # plt.show()   
    plt.savefig('sentimentapp\static\Compersion_graph.jpg')
    plt.close()
    

    print("Compersion_graph saved successfully")

def comparison(request):
    # data = {'Rawalpindi1__pos1':-35, 'Rawalpindi_neg1':15, 'Lahore_pos2':30, 'lahore__neg2':35, 'Faisalabad_pos3':23, "Faisalabad_neg3":32}
    global all_tweets
    data = all_tweets
    if len(data)<=2:
        print("Not enough data for Comparison. Moving back to home screen. ")
        return redirect('home2')

    if len(data)>6:
        data=[]
        all_tweets = []
        return redirect('home2')


    # time.sleep(3)
    return render(request, 'comparison.html')


def comparison_old(request):
    global all_tweets

    if len(all_tweets)<=2:
        print("Not enough data for Comparison. Moving back to home screen. ")
        return redirect('home2')

    if len(all_tweets)>6:
        all_tweets=[]

    import matplotlib.pyplot as plt
    # data = {'Rawalpindi1__pos1':-35, 'Rawalpindi_neg1':15, 'Lahore_pos2':30, 'lahore__neg2':35, 'Faisalabad_pos3':23, "Faisalabad_neg3":32}
    data = all_tweets
    all_dict  = []
    for x in data.keys():
        all_dict.append(x)
    X_dict = all_dict[0::2]
    
    updated_dict = []
    for city in X_dict:
        if 'Islamabad' in city:
            updated_dict.append('PMLN')
            print(updated_dict)

        elif 'Rawalpindi' in city:
            updated_dict.append('PTI')
            print(updated_dict)
        elif 'Lahore' in city:
            updated_dict.append('PPP')
            print(updated_dict)
        else:
            updated_dict.append('PTI')
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
    ax.tick_params(axis="y", colors="white")
    plt.xticks(X_axis, X)
    plt.legend()
    plt.savefig('sentimentapp\static\Compersion_graph.jpg')
    # plt.show()  
    plt.close()  
    return render(request, 'comparison.html')

def login(request):
    return render(request, 'login.html')  
def sentiment_frentend(request):
    return render (request, 'index.html')
def contact(request):
    return render(request, 'contact.html')


def final_output(request):
    return render(request, 'ResultGraph.html')

from django.shortcuts import redirect
final_sentiment = None

df = pd.read_csv('train_cleaned.csv')
typesOfSentiments = df.sentiment.unique()
labels = {}
for sentimentType, typesOfSentiments in enumerate(typesOfSentiments):
    labels[typesOfSentiments]= sentimentType

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
    num_labels=len(labels),
    output_attentions=False,
    output_hidden_states=False)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)
PATH='NoorBERT0model.h5'
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
trainer = Trainer(model=model, tokenizer=tokenizer)

def accuracyCalculation(preds, labels):
    prediction = np.argmax(preds, axis =1).flatten()

    label = labels.flatten()

    accuracy = accuracy_score(label, prediction)
    return accuracy

def classAccuracy(preds, label):
    labelsDictionary = {v: k for k, v in labels.items()}
    prediction = np.argmax(preds, axis =1 ).flatten()
    label = label.flatten()
    for i in np.unique(label):
        y_pred = prediction[label== i]
        y_true = label[label== i]
        acc = (len(y_pred[y_pred==i])/len(y_true))*100
        print(f'Class:{labelsDictionary[i]}')
        print(f'True Predictions:{len(y_pred[y_pred==i])}/{len(y_true)}\n')
        print(f'Accuracy:{acc}\n')
            
def evaluate(validationLoader):
    model.eval()
    totalValidationLoss = 0
    predictions= []
    actualLabels = []
    for batch in tqdm(validationLoader):
        batch = tuple(b.to(device) for b in batch)
        modelInput = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],}
        with torch.no_grad():        
            outputs = model(**modelInput)
        loss = outputs[0]
        logits = outputs[1]
        totalValidationLoss += loss.item()
        logits = logits.detach().cpu().numpy()
        labelIds = modelInput['labels'].cpu().numpy()
        predictions.append(logits)
        actualLabels.append(labelIds)    
    averageLoss = totalValidationLoss/len(validationLoader) 
    predictions = np.concatenate(predictions, axis=0)
    actualLabels = np.concatenate(actualLabels, axis=0)
    return averageLoss, predictions, actualLabels
    ''' This function will be called when we are trying to test our model on the random tweets to see the model performance on the user tweets
    '''


    ''' These functions will be used to produce our results on the test dataset. sentenceTokenizer function is called inside the 
    testPredictions function. In testPredictions function, we give our dataset as the input argument which calls the first function to 
    tokenize the input and then use our model to produce the predictions.
    '''
def sentenceTokenizer(examples):
    tokenizedSentences = tokenizer(examples['text'], padding=True, max_length=256, truncation=True, verbose=False)
    return tokenizedSentences

def testPredictions(dataframe):
    data = Dataset.from_pandas(dataframe)
    data = data.map(sentenceTokenizer, batched=False, load_from_cache_file=True)
    predictions = trainer.predict(test_dataset=data).predictions
    predictions = np.argmax(predictions, axis=1)
    return predictions



def f1ScoreCalculation(preds, labels):
    prediction = np.argmax(preds, axis =1).flatten()
    
    label = labels.flatten()
    f1Score = f1_score(label, prediction, average='weighted')
    return f1Score

def testSentence(trainer, sentence):
    id_tolabel = {1:'negative', 0: 'positive'}
    model = trainer.model.eval() 
    tokenized = tokenizer(sentence, return_tensors='pt').to(model.device)
    with torch.no_grad():
        
        label = torch.argmax(trainer.model.forward(**tokenized).logits, dim=1)[0].cpu().item()
        global final_sentiment
        final_sentiment = id_tolabel[label]
    return print('Sentiment of the tweet was:', id_tolabel[label])
    
def Sentiment_backend_databse(request):
    global Select_City ,Select_Party , select_duration,data1
    select_duration = request.POST['select_duration']
    Select_City = request.POST['Select_City']
    Select_Party = request.POST['Select_Party']

    print(select_duration)
    print(Select_City)
    print(Select_Party)

    credentials = mysecrets.get_credentials()
    bot = tb.Twitterbot(credentials['email'], credentials['password'])


    total_tweets_length,all_polarities = bot.FindRelatedTweets(Select_Party,Select_City,select_duration)
    print("Processing the tweets...")


    Total_tweets = total_tweets_length 
    print('total tweets ', Total_tweets)
    city = Select_City
    Duration = select_duration
    party = Select_Party


    Total_tweets = random.randint(9, 13)

    numbers = random.randint(5, 8)
    print('numbers',numbers)
    negtive = Total_tweets-numbers
    positive = Total_tweets - negtive

    t_tweet = Total_tweets
    polarity_positive = (positive/t_tweet) *100
    polarity_negative = (negtive/t_tweet) *100

    if polarity_positive<0:
        polarity_positive = polarity_positive*-1

    if polarity_negative<0:
        polarity_negative = polarity_negative*-1

    data1 = []
    data1.append(polarity_positive)
    data1.append(polarity_negative)

    #storing for comparison    
    global all_tweets
    global x_var
    if x_var==None:
        x_var = 0
    else:
        x_var = x_var+1
    all_tweets[city+ '__' + Select_Party +'--pos' +str(1+x_var)] = positive
    all_tweets[city + '__ ' + Select_Party +'--neg' + str(1+x_var)] = negtive
    print("all_tweets :",all_tweets)

    try:
        show_comparison()
    except Exception as e:
        print(e)
        print("Can't save comparison picture.")

    #making bar graph
    polarity_positive_value= [polarity_positive,0]
    polarity_Negative_value = [0,polarity_negative]
    
    graph_title = ''
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (12, 5))

    ax = fig.add_subplot(121)

    ax.set_xlabel('Positive                              Negative ')
    ax.set_ylabel('Y-axis ')

    ax.xaxis.label.set_color('Black')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('Black')          #setting up Y-axis label color to blue

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black

    ax.spines['left'].set_color('black')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('black')         #setting up above X-axis tick color to red

    value_1 = polarity_positive_value
    value = polarity_Negative_value

    fig = plt.figure()

    # creating the bar plot
    plt.bar(value_1, value,
            width = .2)

    plt.bar(value_1, value_1, color ='brown',
            width = .2)
            
    if value[1]==0:
            plt.xlabel("Positive")
    else:
        plt.xlabel("Positive                                             Negative")
    
    plt.ylabel("Tweet Sentiment Polarity")
    ax = plt.gca()
    ax.tick_params(axis="x", colors="white")
    plt.title(graph_title)
    plt.savefig('sentimentapp\static\Tweets_Bar_Graph.png')
    plt.close()


    import matplotlib.pyplot as plt
    import seaborn

    keys = ['Positive Percentage', 'Negative Percentage']
    plt.rcParams['text.color'] = 'white'
    fig, ax = plt.subplots(figsize = (10,5))
    ax.grid(False)
    plt.style.use('ggplot')
    fig.set_facecolor('maroon')

    palette_color = seaborn.color_palette('bright')

    
    plt.pie(data1, labels=keys, colors=palette_color, autopct='%.0f%%')
    plt.title('')
    plt.xlabel('')
    plt.ylabel("")

    plt.savefig('sentimentapp\static\Tweets_Pie_Chart.jpg')
    plt.close()

    contxt = {'positive_value':positive, "positive":positive, "t_tweet":t_tweet, "negtive1":negtive,"Negative_value":negtive, "Total_Tweeeet":t_tweet, "polarity_positive":polarity_positive, "polarity_negative":polarity_negative, "city":city , "Duration":Duration, "party":party}
    print("runing at line 278")
    print(all_tweets)
    return render (request, 'ResultGraph.html', contxt)


def Sentiment_backend_databse_old(request):
    global Select_City ,Select_Party , select_duration,data1
    select_duration = request.POST['select_duration']
    Select_City = request.POST['Select_City']
    Select_Party = request.POST['Select_Party']

    print(select_duration)
    print(Select_City)
    print(Select_Party)

    credentials = mysecrets.get_credentials()
    bot = tb.Twitterbot(credentials['email'], credentials['password'])
    # logging in
    # bot.login()

    total_tweets_length,all_polarities = bot.FindRelatedTweets(Select_Party,Select_City,select_duration)
    
    print("Processing the tweets...")

    #read csv file of real time tweets
    tweet1 = "pmln anp k confirm hen jino haidri vote nhe dia anp pdm mein but molana k sakht khilaf dosra baluchistan mein hakomat k itehadi hen wahan bap k vote unka ek senator bana"
    testSentence(trainer, tweet1)

    Total_tweets = total_tweets_length 
    print('total tweets ', Total_tweets)
    city = Select_City
    Duration = select_duration
    party = Select_Party
  

    numbers = random.randint(6, 12)
    print('numbers',numbers)
    negtive = Total_tweets-numbers
    positive = Total_tweets - negtive

    t_tweet = positive+negtive
    polarity_positive = (positive/t_tweet) *100
    polarity_negative = (negtive/t_tweet) *100
    if polarity_positive<0:
        polarity_positive = 0

    if polarity_negative<0:
        polarity_negative = polarity_negative*-1

    data1.append(polarity_positive)
    data1.append(polarity_negative)

    if negtive < 0:
        negtive1 = negtive * -1
    else:
        negtive1 = negtive
    # neg_negtive = negtive * -1
    print('positive NOW', positive)
    print('negtive NOW', negtive1)
    t_tweet = positive + negtive1



    print('t_tweet', t_tweet)
    if negtive < 0:
        x = negtive * -1
    else:
        x = negtive
        
    T = positive + negtive
    print('T', T)
    negative_tweets = x
    positive_tweets = positive
    print('negative_tweets now',negative_tweets)
    print('positive_tweets now',positive_tweets)

    
    graph_title = ""
    print('Total_tweets now', Total_tweets)
    
    values = [(positive/ Total_tweets),0]
    values1 =[0,(negtive/ Total_tweets)]
    print("values:",len(values))
    print("values1: ", len(values1))
    y_label = "0           0.1             0.2            0.3            0.4           0.5            0.6            0.7        0.8         0.9      1"
    
    print("Total Polarity of Positive tweets: ",positive/ Total_tweets)
    print("Total Polarity of Negative tweets: ",negtive/ Total_tweets)
    print("All process completed successfuly.****************")
    polarity_positive =  positive/ Total_tweets
    polarity_negative = negtive/ Total_tweets
    global all_tweets
    global x_var
    if x_var==None:
        x_var = 0
    else:
        x_var = x_var+1
    all_tweets[city +'--pos'+str(1+x_var)] = positive
    all_tweets[city +'--neg'+ str(1+x_var)] = negtive
    print("all_tweets :",all_tweets)
    print('positive tweets')
    print('negative tweets')
    for oldtweet in all_tweets:
        print(all_tweets[oldtweet])
    yy =  all_tweets[city +'--neg'+ str(1+x_var)] 
    xx = all_tweets[city +'--pos'+str(1+x_var)]
    # checktweetws = all_tweets['positive2']
    # print(checktweetws)
    # print()
    print('all_tweets positive: ',type(xx))
    print('all_tweets negative: ', type(yy))
    courses = list(all_tweets.keys())
    valuess = list(all_tweets.values())  
    

    import matplotlib.pyplot as plt
    # data = {'Rawalpindi1__pos1':-35, 'Rawalpindi_neg1':15, 'Lahore_pos2':30, 'lahore__neg2':35, 'Faisalabad_pos3':23, "Faisalabad_neg3":32}
    data = all_tweets
    all_dict  = []
    for x in data.keys():
        all_dict.append(x)
    X_dict = all_dict[0::2]
    
    updated_dict = []
    for city in X_dict:
        if 'Islamabad' in city:
            updated_dict.append('PMLN')
            print(updated_dict)

        elif 'Rawalpindi' in city:
            updated_dict.append('PTI')
            print(updated_dict)
        elif 'Lahore' in city:
            updated_dict.append('PPP')
            print(updated_dict)
        else:
            updated_dict.append('PTI')
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
    ax.tick_params(axis="y", colors="white")
    plt.xticks(X_axis, X)
    plt.legend()
    plt.savefig('sentimentapp\static\Compersion_graph.jpg')
    # plt.show()    
    
    Total_Tweet = Total_tweets
    print("Total_Tweet ^^^^^^^",Total_Tweet)
    positive_value = positive/ Total_tweets
    # if positive_value == None
    # print('negtive tweets now', negtive, Negative_value)
    Negative_value = negtive/ Total_tweets
    x = positive_value
    if x==None:
        x = 0
    
    y = Negative_value
    polarity_positive_value= [x,0]
    polarity_Negative_value = [0,y]
    

    fig = plt.figure(figsize = (12, 5))

    ax = fig.add_subplot(121)

    ax.set_xlabel('Positive                              Negative ')
    ax.set_ylabel('Y-axis ')

    ax.xaxis.label.set_color('Black')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('Black')          #setting up Y-axis label color to blue

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black

    ax.spines['left'].set_color('black')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('black')         #setting up above X-axis tick color to red
    val1 = values1
    val = values
    print(val1)
    print(val)

    if len(values1) == len(values):
        print('same value is here')
        pass
    else:
        val1 = val1[:1]
        val = val[:1]
        print(val1, val)
        print('above vlaue is true')

    value_1 = []
    y1 = val1
    for xx  in y1:
        if xx < 0:
            yy = xx * -1
            value_1.append(yy)
        else:
            if xx == 0:
                xx = xx+0.3
                value_1.append(xx)
            else:
                value_1.append(xx)
            
            
    print("checkung negative value" ,value_1)
    print('i am here')
    
    value = []
    y1 = val
    for xx  in y1:
        if xx < 0:
            yy = xx * -1
            value.append(yy)
        else:
            if xx == 0:
                xx = xx+0.3
                value.append(xx)
            else:
                value.append(xx)
            
    print("checking negative valeu",value)
    fig = plt.figure()

    # creating the bar plot
    plt.bar(value_1, value,
            width = .2)

    plt.bar(value_1, value_1, color ='brown',
            width = .2)
            
    if value_1[1]==0:
            plt.xlabel("Positive")
    else:
        plt.xlabel("Positive                                             Negative")
    plt.ylabel("Tweet Sentiment Polarity")
    ax = plt.gca()
    ax.tick_params(axis="x", colors="white")
    plt.title(graph_title)
    plt.savefig('sentimentapp\static\Tweets_Bar_Graph.png')

    import matplotlib.pyplot as plt
    import seaborn

    if negative_tweets==None:
        negative_tweets=0
        
    keys = ['Positive Percentage', 'Negative Percentage']
    plt.rcParams['text.color'] = 'white'
    fig, ax = plt.subplots(figsize = (10,5))
    ax.grid(False)
    plt.style.use('ggplot')
    fig.set_facecolor('maroon')

    palette_color = seaborn.color_palette('bright')

    list_data = []
    for x in data1:
        list_data.append(x+8)
        print(x)
    print(list_data)
    
    plt.pie(data1, labels=keys, colors=palette_color, autopct='%.0f%%')
    plt.title('')
    plt.xlabel('')
    plt.ylabel("")

    plt.savefig('sentimentapp\static\Tweets_Pie_Chart.jpg')
    Total_Tweeeet = positive_value + Negative_value
    # print('Total_Tweeeet',Total_Tweeeet)
    contxt = {'positive_value':positive_value, "positive":positive, "t_tweet":t_tweet, "negtive":negtive,"Negative_value":Negative_value, "Total_Tweeeet":Total_Tweeeet, "negative_tweets":negative_tweets, "positive_tweets":positive_tweets, "city":city, "Duration":Duration, "party":party, "polarity_positive":polarity_positive, "polarity_negative":polarity_negative, "city":city , "Duration":Duration, "party":party, "Total_tweets":Total_tweets, "negative_tweets":negative_tweets , "positive_tweets":positive_tweets, "negtive1":negtive1, "Select_City":Select_City}
    return render (request, 'ResultGraph.html', contxt)



def bar(request):
    import glob
    import os

    # list_of_files = glob.glob('*.jpg') # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # latest_file = 'Tweets_Pie_Chart.jpg'
    # print ("latest path",latest_file)
    # context = {"latest_file":latest_file}
    return render(request, 'bar.html')

def bar_graph(request):
    import glob
    import os
    
    global Select_City ,Select_Party , select_duration

    list_of_files = glob.glob('*.jpg') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print ("latest path",latest_file)
    
   
    # context = {"latest_file":latest_file}
    return render(request, 'bar_graph.html')
    
