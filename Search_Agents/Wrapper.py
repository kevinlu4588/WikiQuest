from Scraper import get_wikipedia_links_with_similarity
import wikipediaapi
import heapq

def find_path_to_target(start_title, target_title, max_iters=15, similarity_metric="cosine"):
    """
    Find a path from a start Wikipedia page to a target page using cosine similarity.

    :param start_title: Title of the starting Wikipedia page
    :param target_title: Title of the target Wikipedia page
    :param max_iters: Maximum number of iterations (default 6)
    :return: List of page titles representing the path taken or None if not found
    """
    wiki_wiki = wikipediaapi.Wikipedia(user_agent="Wikipedia game solution grapher (flores.r@northeastern.edu)", language="en")
    target_page = wiki_wiki.page(target_title)
    if not target_page.exists():
        return None

    # Priority queue (negative min heap = max heap), contains starting page at first
    priority_queue = [(-1, start_title, [])]  # heap element is: (negative similarity, title, path)

    visited = set()

    for iteration in range(max_iters):
        if not priority_queue:
            break

        # Pop the page with the highest similarity, skipping over those we've already seen
        _, current_title, path = heapq.heappop(priority_queue)
        
        while current_title in visited:
             _, current_title, path = heapq.heappop(priority_queue)


        print(f"For Iteration {iteration}, visiting page: {current_title}")

        visited.add(current_title)

        current_page = wiki_wiki.page(current_title)

        # check if we've reached the target
        if current_title.casefold() == target_title.casefold():
            print("found the target page! Here is the path:")
            return [start_title] + path
        
        if target_title in wiki_wiki.page(current_title).links:
            print("found the target page! Here is the path:")
            return [start_title] + path + [target_title]
        

        # otherwise get all links with their cosine similarities
        links_with_similarity = get_wikipedia_links_with_similarity(current_page, target_page, similarity_metric=similarity_metric)

        # And add the linked pages to the frontier
        for link_title, (similarity, _) in links_with_similarity.items():
            if link_title not in visited:
                new_path = path + [link_title]
                heapq.heappush(priority_queue, (-similarity, link_title, new_path))

    # if we didnt find a path
    print("Unable to find a full path! Here is the path explored thus far:")
    return [start_title] + path



 


if (__name__ == "__main__"):

    paths = find_path_to_target("2005_Azores_subtropical_storm", "Fluid", similarity_metric="cosine") # "cosine" "hamming" "jaccard" "lev"    print("returned path:")
    for i, p in enumerate(paths):
        print(f"step {i}, page: {p}")

    print("------------------------------------------------------------------------------\n" * 4)

