from gensim.models import KeyedVectors
import gensim.downloader
import numpy as np
import re
import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

# Load the pre-trained Word2Vec model
model_path = '/Users/KevinLu/Downloads/wikipedia/model.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
#wiki_vectors = KeyedVectors.load(model_file_path)

word_vector = model['dog_NOUN']

model_file_path = "model.bin"  # Specify the desired file path

wiki_vectors = KeyedVectors.load(model_file_path)
def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

input_word = 'school_NOUN'
user_input = [x.strip() for x in input_word.split(',')]
result_word = []
    
for words in user_input:
    
        sim_words = model.most_similar(words, topn = 5)
        sim_words = append_list(sim_words, words)
            
        result_word.extend(sim_words)
    
similar_word = [word[0] for word in result_word]
similarity = [word[1] for word in result_word] 
similar_word.extend(user_input)
labels = [word[2] for word in result_word]
label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
color_map = [label_dict[x] for x in labels]

import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    
    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    
    for i in range (len(user_input)):

                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0], 
                    y = three_dim[count:count+topn,1],  
                    z = three_dim[count:count+topn,2],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0], 
                    y = three_dim[count:,1],  
                    z = three_dim[count:,2],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
display_pca_scatterplot_3D(model, user_input, similar_word, labels, color_map)

# # Preprocess titles
# def preprocess_title(title):
#     title = title.lower()
#     title = re.sub(r'[()]', '', title)
#     tokens = title.split()
        
#     return tokens

# # Calculate title vectors
# def calculate_title_vector(title, model):
#     tokens = preprocess_title(title)
#     vectors = [model[word] for word in tokens if word in model]
    
#     if vectors:
#         return sum(vectors) / len(vectors)
#     else:
#         return None

# # Calculate cosine similarity between two title vectors
# def calculate_similarity(vector1, vector2):
#     if vector1 is not None and vector2 is not None:
#         return vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
#     else:
#         return 0.0
    

# vector1 = calculate_title_vector("(Apple)", wiki_vectors)
# vector2 = calculate_title_vector("Apple", wiki_vectors)
# print(calculate_similarity(vector1, vector2))

# # print(wiki_vectors.most_similar('dog'))
# # print(wiki_vectors.similarity('dog', 'hello'))
