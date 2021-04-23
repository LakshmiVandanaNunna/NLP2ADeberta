from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import time
import random
import os

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli", num_labels = 3)

best_match = {}
conv =[]
response =''
df = pd.read_csv('datamainlist2.csv',encoding='latin-1')
sdf = pd.read_csv('sample_inputs.csv',encoding='latin-1')
conv=df.conversation
#print("Talk to me: ")
#inpt = input()


for index, row in sdf.iterrows():
    inpt=row["conversation"]
    data = []

    print("Retrieving the response.......")
    score=0
    count=0
    start=time.time()
    # logic to iterate through entire dataset
    for i in range(len(conv)):
        paraphrase = tokenizer.encode_plus(inpt,conv[i], return_tensors="pt")
        paraphrase_classification_logits = model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]

        if(len(best_match)>0 and list(best_match.values())[0]< paraphrase_results[2]):
            best_match = {}
            best_match[i] = paraphrase_results[2]
        elif(len(best_match)==0):
            best_match[i] = paraphrase_results[2]
        count = count+1
        score = score+int(paraphrase_results[2]*100)


    avg_score=score/count
    print("##################################################")
    print("Full Dataset results:")
    print("Average Score for "+ inpt+": "+ str(avg_score))
    response=conv[list(best_match.keys())[0]]
    print('***********************************')
    print('Response:' + response)
    print('***********************************')
    end = time.time()
    data.append([inpt, 'Full Dataset', avg_score, list(best_match.values())[0], end-start ])
    print( 'Entailment score :' + str(list(best_match.values())[0]))
    print('Took ' +str(end-start)+' seconds to process the response with '+str(len(conv))+ ' entries')
    print("##################################################")
    print("                                                  ")


    score=0
    count=0
    best_match = {}
    start=time.time()
    # logic to pick random sentences (500 iterations) from the dataset and pick the best entailment score
    for j in range(0,500):
        i = random.randrange(len(conv))
        paraphrase = tokenizer.encode_plus(inpt, conv[i], return_tensors="pt")
        paraphrase_classification_logits = model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]

        if(len(best_match)>0 and list(best_match.values())[0]< paraphrase_results[2]):
            best_match = {}
            best_match[i] = paraphrase_results[2]
        elif(len(best_match)==0):
            best_match[i] = paraphrase_results[2]
        count = count+1
        score = score+int(paraphrase_results[2]*100)


    avg_score=score/count
    print("##################################################")
    print("Random search for 500 interations")
    print("Average Score for "+ inpt+": "+ str(avg_score))
    response=conv[list(best_match.keys())[0]]
    print('***********************************')
    print('Response:' + response)
    print('***********************************')
    end = time.time()
    data.append([inpt, 'Random search with 500 iterations', avg_score, list(best_match.values())[0], end-start ])
    print( 'Entailment score :' + str(list(best_match.values())[0]))
    print('Took ' +str(end-start)+' seconds to process the response with '+str(len(conv))+ ' entries')
    print("##################################################")
    print("                                                  ")





    score=0
    count=0
    best_match = {}
    start=time.time()
    # logic to iterate on entire dataset but break in case it finds an entailment score higher than the threshold
    for i in range(len(conv)):
        paraphrase = tokenizer.encode_plus(inpt, conv[i], return_tensors="pt")
        paraphrase_classification_logits = model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]

        if(len(best_match)>0 and list(best_match.values())[0]< paraphrase_results[2]):
            best_match = {}
            best_match[i] = paraphrase_results[2]
        elif(len(best_match)==0):
            best_match[i] = paraphrase_results[2]
        count = count+1
        score = score+int(paraphrase_results[2]*100)
        if(int(paraphrase_results[2]) > 50):
            break
    avg_score=score/count
    print("##################################################")
    print("Full Dataset with a threshold limit:")
    print("Average Score for "+ inpt+": "+ str(avg_score))
    response=conv[list(best_match.keys())[0]]
    print('***********************************')
    print('Response:' + response)
    print('***********************************')
    end = time.time()
    data.append([inpt, 'Full Dataset with Threshold at 50', avg_score, list(best_match.values())[0], end-start ])
    print( 'Entailment score :' + str(list(best_match.values())[0]))
    print('Took ' +str(end-start)+' seconds to process the response with '+str(len(conv))+ ' entries')
    print("##################################################")
    print("                                                  ")



    df = pd.DataFrame(data, columns=["Input", "Type", "avg_score", "entailment_score", "time_taken"])

    if not os.path.isfile('metrics.csv'):
       df.to_csv('metrics.csv', header=["Input", "Type", "avg_score", "entailment_score", "time_taken" ])
    else:
       df.to_csv('metrics.csv', mode='a', header=False)

import pandas as pd

entailments = pd.read_csv('metrics.csv')

