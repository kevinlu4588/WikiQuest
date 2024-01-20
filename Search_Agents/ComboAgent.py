from Scraperv3 import word_vec_get_wikipedia_links_with_similarity
import wikipediaapi
import heapq
import Search_Agents.util as util
import random
import copy

def find_path_to_target(start_title, target_title, max_iters=50, epsilon_explore = True, epsilon = 0.1):
    print("Start: " + start_title)
    print("Target: " + target_title)
    """
    Find a path from a start Wikipedia page to a target page using cosine similarity.

    :param start_title: Title of the starting Wikipedia page
    :param target_title: Title of the target Wikipedia page
    :param max_iters: Maximum number of iterations (default 6)
    :return: List of page titles representing the path taken or None if not found
    """

    # Priority queue (negative min heap = max heap), contains starting page at first
    priority_queue = [(-1, start_title, [])]  # heap element is: (negative similarity, title, path)

    visited = set()

    for iteration in range(max_iters):
        if not priority_queue:
            break
        if(epsilon_explore):
            #Epsilon Greedy Implementation
            random_heap = copy.copy(priority_queue)
            random.shuffle(random_heap)
            if util.flipCoin(epsilon):
                # Pop the page with the highest similarity, skipping over those we've already seen
                _, current_title, path = heapq.heappop(random_heap)
            else:
            # Pop the page with the highest similarity, skipping over those we've already seen
                _, current_title, path = heapq.heappop(priority_queue)
        else:
            # Pop the page with the highest similarity, skipping over those we've already seen
                _, current_title, path = heapq.heappop(priority_queue)


        # # Pop the page with the highest similarity, skipping over those we've already seen
        # _, current_title, path = heapq.heappop(priority_queue)
        
        while current_title in visited:
             _, current_title, path = heapq.heappop(priority_queue)


        print(f"For Iteration {iteration}, visiting page: {current_title}")

        visited.add(current_title)

        # check if we've reached the target
        if current_title.casefold() == target_title.casefold():
            print("found the target page! Here is the path:")
            return priority_queue, [start_title] + path

        # otherwise get all links with their cosine similarities
        links_with_similarity = word_vec_get_wikipedia_links_with_similarity(current_title, target_title)

        # And add the linked pages to the frontier
        for link_title, (similarity, _) in links_with_similarity.items():
            if link_title not in visited:
                new_path = path + [link_title]
                heapq.heappush(priority_queue, (-similarity, link_title, new_path))

    # if we didnt find a path
    print("Unable to find a full path! Here is the path explored thus far:")
    return priority_queue, [start_title] + path


        

if (__name__ == "__main__"):

    _, paths = find_path_to_target("Mount Everest", "Leonardo da Vinci")
    print("returned path:")
    for i, p in enumerate(paths):
        print(f"step {i}, page: {p}")
