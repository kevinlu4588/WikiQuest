import matplotlib.pyplot as plt
import numpy as np
import pickle
# Sample data
approaches = ['DLS Epsilon Greedy']  # Replace with your approach names
num_runs = 100
file_path = 'my_tfidf.pkl'

# Loading the data from the pickle file
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)


time_taken_data, priority_queue_data, accuracy_data = loaded_data


print(np.around(priority_queue_data, 0))
print(time_taken_data)
print(accuracy_data)
