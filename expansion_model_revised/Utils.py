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
stopwords = stopwords.words()
lemmatizer = WordNetLemmatizer()
import os

def preprocess(text):
    words = word_tokenize(text.lower())
    #lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        if word not in stopwords:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)

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
            if rel_jud[tup[0]]>0:
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
    scores= {"ndcg1": ndcg1, "ndcg3": ndcg3, "ndcg5": ndcg5, "ndcg10": ndcg10, "ndcg100": ndcg100, "p5": p5/5.0,
            "p10": p10/10.0, "p1": p1/1.0, "p3": p3/3.0, "p100": p100/100.0, "num_rel": num_rel}
    scores.update(get_ap(rel_jud,sorted_res))
    return scores


def p_at_k(sorted_res,rel_jud,k):
    p=0
    for rank,tup in enumerate(sorted_res[:k]):
        try:
                if rel_jud[tup[0]]>0:
                    p+=1
        except:
            continue
    return float(p)/float(k)

def get_ap(rel_jud,sorted_res):
    ap1_q = 0.0
    ap5_q = 0.0
    ap3_q=0.0
    ap10_q = 0.0
    ap100_q = 0.0
    ap_full_q = 0.0
    for rank, tup in enumerate(sorted_res):
        try:
            if rel_jud[tup[0]]>0:
                p_rank = p_at_k(sorted_res,rel_jud,rank+1)
                if rank<1:
                    ap1_q += p_rank
                if rank < 3:
                    ap3_q += p_rank
                if rank < 5:
                    ap5_q += p_rank
                if rank < 10:
                    ap10_q += p_rank
                if rank < 100:
                    ap100_q += p_rank
                ap_full_q += p_rank
        except:
            continue

    ap1_q = float(ap1_q)/float(len(rel_jud))
    ap3_q = float(ap3_q)/float(len(rel_jud))
    ap5_q = float(ap5_q)/float(len(rel_jud))
    ap10_q = float(ap10_q)/float(len(rel_jud))
    ap100_q = float(ap100_q)/float(len(rel_jud))
    ap_full_q = float(ap_full_q)/float(len(rel_jud))
    return {"ap1":ap1_q,"ap3":ap3_q,"ap5":ap5_q,"ap10":ap10_q,"ap100":ap100_q,"ap_full":ap_full_q}


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
    ideal_sorted = sorted(ideal_sorted.items(), key=operator.itemgetter(1), reverse=True)

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

def write_scores(all_sorted_res,all_rel_jud,results_file):
    res = []
    for i,sorted_res in enumerate(all_sorted_res):
        res.append(evaluate_res(sorted_res,all_rel_jud[i]))
    res_keys = ['p1','p3','p5','p10','p100','ndcg1','ndcg3','ndcg5','ndcg10','ndcg100','ap1','ap3','ap5','ap10','ap100','ap_full','num_rel']

    avgs = [0]*len(res_keys)
    with open(results_file,'w') as f:
        for k in res_keys:
            f.write(k+',')
        f.write('\n')
        for r in res:
            for i,k in enumerate(res_keys):
                avgs[i] += r[k]
                f.write(str(r[k])+',')
            f.write('\n')
        for a in avgs:
            f.write(str(a/len(all_sorted_res))+',')

if __name__ == '__main__':
    '''
    true_pos_dir = '/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/true_positives'
    pred_dir= '/Users/bhavya/Documents/cs512/project/data/results/bm25_predictions'
    results_file = '/Users/bhavya/Documents/cs512/project/data/results/bm25_all_scores.txt'
    q_files = os.listdir(pred_dir)
    all_sorted_res = []
    all_rel_jud = []
    for qf in q_files:
        with open(os.path.join(true_pos_dir,qf),'r') as f:
            rel_jud = {}
            tps = [tp.strip('\n') for tp in f.readlines()]
            for tp in tps:
                rel_jud[tp] = 1
            all_rel_jud.append(rel_jud)
        with open(os.path.join(pred_dir,qf),'r') as f:
            sorted_res = []
            preds = f.readlines()
            for p in preds:
                sorted_res.append(tuple(p.split()))
            all_sorted_res.append(sorted_res)
    write_scores(all_sorted_res,all_rel_jud,results_file)
    '''
            


    
