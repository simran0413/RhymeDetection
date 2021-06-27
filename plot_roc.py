import json
import matplotlib.pyplot as plt

f = open('results/roc_results_full_balanced.json')
data = json.load(f)

g = open('results/roc_results_full_balanced.json')
data2 = json.load(g)


fig = plt.figure(figsize=(10,10))
for i in data["results"]:
    fpr = i["fpr"]
    tpr = i["tpr"]
    plt.plot(fpr, tpr, label = i["type"])
for i in data2["results"]:
    fpr = i["fpr"]
    tpr = i["tpr"]
    plt.plot(fpr, tpr, label = i["type"])
plt.legend()
plt.show()

plt.savefig("results/roc_balanced.png", bbox_inches = 'tight')