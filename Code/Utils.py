import os
import operator
import numpy as np
import math
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import json
import ast
import operator
import pickle
import io    
import random
import ast
import string
from sklearn.preprocessing import MinMaxScaler
import re
stopwords = stopwords.words()
stopwords = dict(zip(stopwords, range(len(stopwords))))
lemmatizer = WordNetLemmatizer()
lemmatized_words = {}
#table = {char: None for char in string.punctuation}
#table = string.maketrans("", "", string.punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))
def preprocess(text):
    text = regex.sub('', text)
    words = text.lower().split()
    #lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        #if word not in stopwords:
        try:
            lemmas.append(lemmatized_words[word])
        except:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
            lemmatized_words[word] = lemma

    text = ' '.join(lemmas)
    return text



def evaluate_res(sorted_res, rel_jud):
    p5 = 0
    p10 = 0
    p1 = 0
    p3 = 0
    num_rel = 0
    p100 = 0
    for rank, tup in enumerate(sorted_res):
        try:
            if rel_jud[tup[0]] > 0:
                if rank < 1:
                    p1 += 1
                if rank < 3:
                    p3 += 1
                if rank < 5:
                    p5 += 1
                if rank < 10:
                    p10 += 1
                if rank < 100:
                    p100 += 1
                num_rel += 1
        except:
            continue

    ndcg1 = get_ndcg(sorted_res, rel_jud, 1)
    ndcg3 = get_ndcg(sorted_res, rel_jud, 3)
    ndcg5 = get_ndcg(sorted_res, rel_jud, 5)
    ndcg10 = get_ndcg(sorted_res, rel_jud, 10)
    ndcg100 = get_ndcg(sorted_res, rel_jud, 100)
    return {"ndcg1": ndcg1, "ndcg3": ndcg3, "ndcg5": ndcg5, "ndcg10": ndcg10, "ndcg100": ndcg100, "p5": p5/5.0,
            "p10": p10/10.0, "p1": p1/1.0, "p3": p3/3.0, "p100": p100/100.0, "num_rel": num_rel}


def get_ndcg(sorted_res, rel_jud, cutoff):
    dcg = 0
    '''
    print (rel_jud.keys())
    for i in range(min(cutoff, len(sorted_res))):
        doc_id = sorted_res[i][0]
        if doc_id not in rel_jud.keys():
            rel_level = 0
        else: 
            rel_level = rel_jud[doc_id]
        print (doc_id, rel_level)
        dcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))
    '''
    for i in range(min(cutoff, len(sorted_res))):
        doc_id = sorted_res[i][0]
        if doc_id in rel_jud:
            rel_level = rel_jud[doc_id]
        else:
            rel_level = 0
        dcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))

    ideal_sorted = {}
    for tup in sorted_res:
        try:
            ideal_sorted[tup[0]] = rel_jud[tup[0]]
        except:
            ideal_sorted[tup[0]] = 0
    ideal_sorted = sorted(ideal_sorted.iteritems(), key=operator.itemgetter(1), reverse=True)

    idcg = 0
    for i in range(min(cutoff, len(ideal_sorted))):
        doc_id = ideal_sorted[i][0]
        try:
            rel_level = rel_jud[doc_id]
        except:
            rel_level = 0
        idcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))
    if idcg == 0:
        idcg = 1

    return dcg/idcg
