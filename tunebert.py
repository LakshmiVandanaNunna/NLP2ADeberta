from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli", num_labels = 3)

best_match = {}
conv =[]
response =''
df = pd.read_csv('datamainlist2.csv',encoding='latin-1')

conv=df.conversation
print("Talk to me: ")
inpt = input()
start=time.time()
print("Retrieving the response.......")
for i in range(len(conv)):
    paraphrase = tokenizer.encode_plus(inpt,conv[i], return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase)[0]
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    #print('Entailment score with '+str(i)+' is ' + str(paraphrase_results[2]) )
    if(len(best_match)>0 and list(best_match.values())[0]< paraphrase_results[2]):
        best_match = {}
        best_match[i] = paraphrase_results[2]
    elif(len(best_match)==0):
        best_match[i] = paraphrase_results[2]

response=conv[list(best_match.keys())[0]]
print('***********************************')
print('Response:' + response)
print('***********************************')
end = time.time()
print( 'Entailment score :' + str(list(best_match.values())[0]))
print('Took ' +str(end-start)+' seconds to process the response with '+str(len(conv))+ ' entries')