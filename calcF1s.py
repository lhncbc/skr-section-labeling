import sklearn.metrics
import numpy as np

labels = ['objective', 'background', 'methods', 'results', 'conclusions']
avg = 'macro'

preds = []
with open('predictions.txt', 'r') as f:
    for line in f:
        preds.append(line.replace('__label__', '').replace('\n', ''))

print(len(preds))

targets = []
with open('/nfsvol/crfiler-skr/Zeshan/structure abstract dataset/test.txt', 'r') as f:
    for line in f:
        tokens = line.replace('\n', '').split('|')
        label = tokens[6].lower()
        if(label != 'none'):
            targets.append(label)
print(len(targets))

preds = np.array(preds)
targets = np.array(targets)
f1s = sklearn.metrics.f1_score(targets, preds, average=None, labels=labels)
precisions = sklearn.metrics.precision_score(targets, preds, average=None, labels=labels)
recalls = sklearn.metrics.recall_score(targets, preds, average=None, labels=labels)
for i in range(5):
    print("%d: f1: %.2f pre: %.2f re: %.2f" % (i, f1s[i], precisions[i], recalls[i]))

f1 = sklearn.metrics.f1_score(targets, preds, average=avg)
precision = sklearn.metrics.precision_score(targets, preds, average=avg)
recall = sklearn.metrics.recall_score(targets, preds, average=avg)
acc = sklearn.metrics.accuracy_score(targets, preds)
print("Overall: f1: %.2f pre: %.2f re: %.2f" % (f1, precision, recall))
