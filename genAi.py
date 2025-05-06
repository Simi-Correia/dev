from gensim.models import Word2Vec

sentences = [["this", "phone", "is", "great"], 
             ["this", "phone", "is", "awful"], 
             ["the", "battery", "life", "is", "amazing"], 
             ["the", "battery", "dies", "quickly"]]
# here we have provided 
# sentences as our train data to word2vec
# then we pass vector_size it turns each word into list of 50 numbers. small dataset so used 50
# then we pass window as max distance between target word and surrounding word 
# min_count is number of time the word must appear and in our case dataset is tiny so passed just 1
# sg is "skip-gram" decides which algo the model uses to learn 
# here sg=0 is CBOW
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0)

# on the model we use the most_similar and pass the word and topn ie it return top 4 most similar to great
similar_words = model.wv.most_similar("great", topn=4)
print("words similar to 'great':", similar_words)

