import matplotlib.pyplot as plt
import numpy as np
import pickle
# Sample data
approaches = ['No Epsilon WordVec', 'TFIDF Vectorizer', 'Combo Agent']  # Replace with your approach names
num_runs = 100
file_path = 'no_greedy_DLS.pkl'

# Loading the data from the pickle file
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

with open('my_tfidf.pkl', 'rb') as file1:
    tfidf_data = pickle.load(file1)


with open('combo_agent.pkl', 'rb') as file2:
    combo_agent = pickle.load(file2)

time_taken_data, path_data, priority_queue_data, accuracy_data = loaded_data
print(type(loaded_data))
tfidf_time, tfidf_path_data, tfidf_priority_queue, tfidf_accuracy_data = tfidf_data

for i in range(len(loaded_data[1])):
    loaded_data[1][i] = len(loaded_data[1][i])
#loaded_data[1] = len(path_data)

for i in range(len(tfidf_data[1])):
    tfidf_data[1][i] = len(eval(tfidf_data[1][i]))
for i in range(len(combo_agent[1])):
    combo_agent[1][i] = len(combo_agent[1][i])


plotted_data = [loaded_data, tfidf_data, combo_agent]

# Plotting histograms
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))

bar_width = 0.7
for i, metric in enumerate(['Time Taken(Seconds)', 'Average Path Length', 'Priority Queue Size', 'Accuracy']):
    if (i == 3):
        axes[i].set_ylim(0.75, 1)  # Set your desired y-axis scale
    axes[i].bar(np.arange(len(approaches)) - bar_width, [np.mean(plotted_data[0][i]), np.mean(plotted_data[1][i]), np.mean(plotted_data[2][i])],
            width=bar_width, label=metric, align='center', color=['blue', 'orange'])


    axes[i].set_xticks(np.arange(len(approaches))- bar_width)
    axes[i].set_xticklabels(approaches)
    axes[i].set_ylabel(metric)
    

# Adding legend outside the loop

plt.tight_layout()
plt.show()
