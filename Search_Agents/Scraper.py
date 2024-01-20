import wikipediaapi
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.spatial.distance import hamming, jaccard
import Levenshtein




def get_wikipedia_links_with_similarity(input_data, reference_page : wikipediaapi.WikipediaPage, similarity_metric="cosine"):
    """
    Get Wikipedia page links with cosine similarity to a reference page.

    :param input_data: URL, title of the Wikipedia page, or WikipediaPage object
    :param reference_page: WikipediaPage object to compare with
    :return: Dictionary with link text as keys and a list [cosine_similarity, wikipedia_page] as values
    """
    wiki_wiki = wikipediaapi.Wikipedia(user_agent="Wikipedia game solution grapher (flores.r@northeastern.edu)",language="en")

    # Determine the type of input_data and get the page
    if isinstance(input_data, wikipediaapi.WikipediaPage):
        page = input_data
    else:
        if input_data.startswith("https://") or input_data.startswith("http://"):
            match = re.search(r"/wiki/(.+)", input_data)
            if match:
                page_title = match.group(1).replace("_", " ")
            else:
                return {"Error": "Invalid Wikipedia URL"}
        else:
            page_title = input_data

        page = wiki_wiki.page(page_title)
        if not page.exists():
            return {"Error": "Page does not exist"}

    # Collect all the pages, making the server requests. This is the most time-intensive part
    # it runs slightly above 12 requests per second
    texts = [reference_page.text]
    offshoot_pages = []
    valid_links = [l for l in page.links.values() if l.namespace == 0 and l.exists]
    for link in tqdm(valid_links, desc="Collecting links..."):
        offshoot_page = wiki_wiki.page(link.title)
        texts.append(offshoot_page.text)
        offshoot_pages.append(offshoot_page)

    

    
    # Calculate cosine similarities
    similarities = {}
    
    

    if similarity_metric == "cosine":
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        ref_vector = tfidf_matrix[0]
        for i, offshoot_page in enumerate(tqdm(offshoot_pages, desc="calculating cosine similarities..."), start=1):
            similarity = cosine_similarity(ref_vector, tfidf_matrix[i])[0][0]
            similarities[offshoot_page.title] = [similarity, offshoot_page]

    elif similarity_metric == "hamming":
        for i, offshoot_page in enumerate(tqdm(offshoot_pages, desc="calculating hamming similarities..."), start=1):
            similarity = calculate_hamming_similarity(reference_page.text, offshoot_page.text)
            similarities[offshoot_page.title] = [similarity, offshoot_page]

    elif similarity_metric == "jaccard":
        for i, offshoot_page in enumerate(tqdm(offshoot_pages, desc="calculating jaccard similarities..."), start=1):
            similarity = calculate_jaccard_similarity(reference_page.text, offshoot_page.text)
            similarities[offshoot_page.title] = [similarity, offshoot_page]

    elif similarity_metric == "lev": # levenshtein
        for i, offshoot_page in enumerate(tqdm(offshoot_pages, desc="calculating levenshtein similarities..."), start=1):
            similarity = calculate_levenshtein_similarity(reference_page.text, offshoot_page.text)
            similarities[offshoot_page.title] = [similarity, offshoot_page]

    else:
        return {"Error" : f"Similarity metric {similarity_metric} not found"} 

    return similarities


vectorizer = TfidfVectorizer()

def text_to_tfidf(text):
    tfidf_vector = vectorizer.fit_transform([text])
    return tfidf_vector



# def calculate_cosine_similarity(tfidf_vector1, tfidf_vector2):
#     similarity = cosine_similarity(tfidf_vector1, tfidf_vector2)
#     return similarity[0][0]


# Disance metrics:
# all of them return 1 - distance, so that the lower the distance the higher the similarity
def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return 1 - jaccard(set1, set2)

def calculate_levenshtein_similarity(text1, text2):
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1
    distance = Levenshtein.distance(text1, text2)
    return 1 - (distance / max_len)


def calculate_hamming_similarity(text1, text2):
    max_len = max(len(text1), len(text2))
    text1 = text1.ljust(max_len)
    text2 = text2.ljust(max_len)
    return 1 - hamming(list(text1), list(text2))


# Test case
if (__name__ == "__main__"):
    w = wikipediaapi.Wikipedia(user_agent="Wikipedia game solution grapher (flores.r@northeastern.edu)",language="en")
    target = w.page("Candle")
    links = get_wikipedia_links_with_similarity("Apple", target)
    print(links)
