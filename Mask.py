import random
from Search_Agents.ComboAgent import find_path_to_target
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import copy
import wikipediaapi 
import numpy as np
def calc_page_similarity(page_1_text, page_2_text):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    #hllo
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform([page_1_text, page_2_text])

    # Calculate the cosine similarity between the two text vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity

def get_page_text(page_title):
        wiki_wiki = wikipediaapi.Wikipedia(user_agent="Wikipedia game solution grapher (lu.kev@northeastern.edu)",language="en")
        page = wiki_wiki.page(page_title)
        if not page.exists():
            return {"Error": "Page does not exist"}
        
        return page.text
#Mask to run wrappers with different parameters
def run_search_program(word_bag, num_runs):
    start_time = time.time()
    times = []
    priority_queue_sizes = []
    paths = []
    results = []
    print("--- %s seconds ---" % (time.time() - start_time))
    for i in range(num_runs):
        print("iteration number: " + str(i))
        copy_of_word_bag = copy.deepcopy(word_bag)

        words = random.sample(copy_of_word_bag, 2)
        for item in words:
            copy_of_word_bag.remove(item)
        try:
            priority_queue, path = find_path_to_target(words[0], words[1], 20, True, 0.1)
        except Exception as e:
            # Print the exception error message
            print(f"Exception: {e}")
            continue
        runtime = time.time() - start_time
        final_word = path[-1]
        similarity = calc_page_similarity(get_page_text(words[1]), get_page_text(final_word))
        start_time = time.time()
        times.append(runtime)
        paths.append(path)
        print(priority_queue)
        print(len(priority_queue))
        priority_queue_sizes.append(len(priority_queue))
        results.append(similarity)
        # Save variables to a file using pickle.dump
    with open('combo_agent.pkl', 'wb') as file:
        pickle.dump((times, paths, priority_queue_sizes, results), file)
        print(paths)
        print(results)

#Creates bag of words
def get_words(file_path):
    # Specify the path to your text file

    # Read the titles from the text file into a list
    with open(file_path, 'r') as file:
        titles_list = [line.strip() for line in file]
    return titles_list

if (__name__ == "__main__"):
    # file_path = 'wikipedia_articles.txt'
    # word_bag = get_words(file_path)
    # run_search_program(word_bag, 20)

    results = [0.75, 1.0, 0.99, 1.0, 1.0000000000000002, 0.6139397697441861, 1.0, 1.0, 1.0, 1.0000000000000002, 1.0, 1.0, 1.0000000000000002, 1.0, 1.0, 1.0, 1.0]

    with open('combo_agent.pkl', 'rb') as file1:
        tfidf_data = pickle.load(file1)

    time_taken_data, path_data, priority_queue_data, accuracy_data = tfidf_data


    with open('combo_agent.pkl', 'wb') as file:
        pickle.dump((time_taken_data, path_data, priority_queue_data, results), file)
        print(results)
        print(np.mean(results))
        print("smh")
    
    