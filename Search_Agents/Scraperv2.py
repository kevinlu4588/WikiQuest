import requests
import re
from gensim.models import KeyedVectors
import gensim.downloader
import numpy as np
import re
from tqdm import tqdm


def word_vec_get_wikipedia_links_with_similarity(input_data, target_title):
    if input_data.startswith("https://") or input_data.startswith("http://"):
        match = re.search(r"/wiki/(.+)", input_data)
        if match:
            page_title = match.group(1).replace("_", " ")
        else:
            return {"Error": "Invalid Wikipedia URL"}
    else:
        page_title = input_data

    ENDPOINT = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "links",
        "pllimit": "max"
    }

    try:
        response = requests.get(ENDPOINT, params=params)
        data = response.json()

        if 'error' in data:
            return {"Error": "Page does not exist"}


        links = {}
        for link in data['parse']['links']:
            if link.get('ns') == 0 and 'exists' in link:
                link_title = link['*']  
                link_url = "https://en.wikipedia.org/wiki/" + link_title.replace(' ', '_')
                links[link_title] = link_url

        #Path to gensim model
        model_file_path = "model.bin"  # Specify the desired file path
        #Using gensim pre-trained model(glove-wiki-gigaword-50)
        wiki_vectors = KeyedVectors.load(model_file_path)
        refVector = calculate_title_vector(target_title, wiki_vectors)
        similiarities ={}
        for next_page_title in links.keys():
            keyVector = calculate_title_vector(next_page_title, wiki_vectors)
            similarity = calculate_similarity(keyVector, refVector)
            similiarities[next_page_title] = [similarity, links.get(next_page_title)]
        return similiarities
    except Exception as e:
        return {"error occured": str(e)}

#Process page titles => lowercase + remove all parentheses
#Returns list of words in title(tokens)
def preprocess_title(title):
    title = title.lower()
    title = re.sub(r'[()]', '', title)
    tokens = title.split()
        
    return tokens

# Calculate average vector representation of words in title
# Takes in a pretrained gensim model
def calculate_title_vector(title, model):
    tokens = preprocess_title(title)
   
    vectors = [model[word] for word in tokens if word in model]
    
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

# Calculate cosine similarity between two title vectors
def calculate_similarity(vector1, vector2):
    if vector1 is not None and vector2 is not None:
        return vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    else:
        return 0.0




# Test case
if (__name__ == "__main__"):
   # Test case
    input_data = "Apple"
    links = word_vec_get_wikipedia_links_with_similarity(input_data, "Candle")
    print(links)
   # print(links.items())

