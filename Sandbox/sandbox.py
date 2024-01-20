import matplotlib.pyplot as plt
import numpy as np
import pickle

approaches = ['WordVec Approach', 'TFIDF Vectorizer Approach']  # Replace with your approach names

# Loading the data from the pickle file
with open("no_greedy_DLS.pkl", 'rb') as file:
    loaded_data = pickle.load(file)

with open('my_tfidf.pkl', 'rb') as file1:
    tfidf_data = pickle.load(file1)

time_taken_data, path_data, priority_queue_data, accuracy_data = loaded_data
for i in range(len(loaded_data[1])):
    loaded_data[1][i] = len(loaded_data[1][i])
#loaded_data[1] = len(path_data)

tfidf_time, tfidf_path_data, tfidf_priority_queue, tfidf_accuracy_data = tfidf_data
for i in range(len(tfidf_data[1])):
    tfidf_data[1][i] = len(eval(tfidf_data[1][i]))

print(tfidf_data[1])
data = [loaded_data, tfidf_data]
data1 = np.concatenate((loaded_data,tfidf_data),axis=1)

def min_max_scaling(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) 

for i in range(len(data)):
    for index in range(len(data[i])):
        min_max_scaling(data[i][index], np.min(data1[index], axis = 0), np.min(data1[index], axis = 0))
# Plotting histograms
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))


for i, metric in enumerate(['Approaches']):
    axes.bar(np.arange(len(approaches)) - 0.2, data[i],
                width=0.2, label='Time Taken', align='center')
    axes.bar(np.arange(len(approaches)), data[i],
                width=0.2, label='Priority Queue Size', align='center')
    axes.bar(np.arange(len(approaches)) + 0.2, data[i],
                width=0.2, label='Accuracy', align='center')

    axes.set_xticks(np.arange(len(approaches)))
    axes.set_xticklabels(approaches)
    axes.set_ylabel(metric)