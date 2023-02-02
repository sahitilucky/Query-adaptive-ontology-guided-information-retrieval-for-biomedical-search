#get_idf
from Utils import *
corpus = {}
with open('../data_files/bioconcepts2pubtator_corpus.json', 'r') as infile:
    corpus = json.load(infile)
'''
context_word_vocabulary = {}
with open('../LC_related_data/context_word_vocabulary.json', 'r') as infile:
    context_word_vocabulary = json.load(infile)
'''

total_docs = len(corpus.keys())
print (total_docs)
#print (len(context_word_vocabulary.keys()))

corpus_words = {}
lemmatized_words = {}
i = 0
word_idf = {}
for doc in corpus:
    text = corpus[doc]
    #print text
    words = list(set(text.lower().split(' ')))
    #print words
    words =  preprocess(' '.join(words)).split(' ')
    #corpus_words[doc] = words
    #print (words)
    for word in words:
        try:
            word_idf[word] += 1
        except:
            word_idf[word] = 1
            '''
            lemma = word
            try:
                lemma = lemmatized_words[word]
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemmatized_words[word] = lemma
            if lemma in context_word_vocabulary:
                word_idf[lemma] += 1
            '''
    i +=1
    if (i%10000)==0:
        print (i)
    if (i%200000)==0:
        print ('Saving intermmediate result')
        print (len(word_idf.keys()))
        with open('../data_files/bioconcepts2pubtator_corpus_idfs_temp.json', 'w') as outfile:
            json.dump(word_idf,outfile)

for word in word_idf:
    if word_idf[word] == 0:
        print ('omg')
    word_idf[word] = math.log(float(total_docs)/float(word_idf[word]+1))

print (len(word_idf.keys()))
with open('../data_files/bioconcepts2pubtator_corpus_idfs.json', 'w') as outfile:
    json.dump(word_idf,outfile)