import matplotlib.pyplot as plt
import numpy as np
import pickle
# Sample data
approaches = ['WordVec No Epsilon', 'WordVec Epsilon']  # Replace with your approach names
num_runs = 100
file_path = 'no_greedy_DLS.pkl'

# Loading the data from the pickle file
with open("no_greedy_DLS.pkl", 'rb') as file:
    loaded_data = pickle.load(file)

with open('greedy_DLS.pkl', 'rb') as file1:
    tfidf_data = pickle.load(file1)

time_taken_data, path_data, priority_queue_data, accuracy_data = loaded_data
for i in range(len(loaded_data[1])):
    loaded_data[1][i] = len(loaded_data[1][i])
#loaded_data[1] = len(path_data)

tfidf_time, tfidf_path_data, tfidf_priority_queue, tfidf_accuracy_data = tfidf_data
for i in range(len(tfidf_data[1])):
    tfidf_data[1][i] = len(tfidf_data[1][i])

## Sample data for time taken, size of priority queue, and accuracy for each approach
#time_taken_data = np.random.rand(num_runs, 3) * 10  # Replace with your actual time taken data
#priority_queue_data = np.random.randint(1, 100, size=(num_runs, 3))  # Replace with your actual priority queue size data
#accuracy_data = np.random.rand(num_runs, 3)  # Replace with your actual accuracy data

# Min-Max scaling function
def min_max_scaling(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val) 

time_taken_data = np.mean(min_max_scaling(time_taken_data))
priority_queue_data = np.mean(min_max_scaling(priority_queue_data))
accuracy_data = np.mean(min_max_scaling(accuracy_data))

tfidf_time = np.mean(min_max_scaling(tfidf_time))
tfidf_priority_queue = np.mean(min_max_scaling(tfidf_priority_queue))
tfidf_accuracy_data = np.mean(min_max_scaling(tfidf_accuracy_data))



# Plotting histograms
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

for i, metric in enumerate(['Approaches']):
    axes.bar(np.arange(len(approaches)) - 0.2, [time_taken_data, tfidf_time],
                width=0.2, label='Time Taken', align='center')
    axes.bar(np.arange(len(approaches)), [priority_queue_data, tfidf_priority_queue],
                width=0.2, label='Priority Queue Size', align='center')
    axes.bar(np.arange(len(approaches)) + 0.2, [accuracy_data, tfidf_accuracy_data],
                width=0.2, label='Accuracy', align='center')

    axes.set_xticks(np.arange(len(approaches)))
    axes.set_xticklabels(approaches)
    axes.set_ylabel(metric)

axes.legend()
plt.tight_layout()
plt.show()