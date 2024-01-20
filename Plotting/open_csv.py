import pandas
import random
from Search_Agents.WordVecWrapper import find_path_to_target
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append("/Users/KevinLu/Desktop/Wikipedia-Game-Solver-CS4100")

import wikipediaapi
def get_page_text(page_title):
        wiki_wiki = wikipediaapi.Wikipedia(user_agent="Wikipedia game solution grapher (lu.kev@northeastern.edu)",language="en")
        page = wiki_wiki.page(page_title)
        if not page.exists():
            return {"Error": "Page does not exist"}
        
        return page.text

def calc_page_similarity(page_1_text, page_2_text):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform([page_1_text, page_2_text])

    # Calculate the cosine similarity between the two text vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity

csvFile = pandas.read_csv('25Results.csv')
runtimes = csvFile['runtimes']
priority_queue_sizes = []
finished_or_not = csvFile['finished?']
similarities = []
paths = csvFile['pages visited']
for times in runtimes:
     priority_queue_sizes.append(times * 1)

for index, row in csvFile.iterrows():
     if row['finished?']:  
          similarities.append(1.00)
     else:
          path = eval(row['pages visited'])
          target_page = get_page_text('bitcoin')
          end_page = get_page_text(path[-1])
          similarity = calc_page_similarity(target_page, end_page)
          similarities.append(similarity)
print(similarities)


with open('my_tfidf.pkl', 'wb') as file:
        pickle.dump((runtimes, paths, priority_queue_sizes, similarities), file)
          


