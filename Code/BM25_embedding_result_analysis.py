from Utils import *

min_bm25_score = 9999999
min_mesh_score = 9999999
ranker_scores = {}
for qid in range(50):
    ranker_scores[qid] = {}
    with open('../BM25_results/predictions/'+ str(qid+1) + '.txt' ,'r') as infile:
        for line in infile:
            doc_num = line.strip().split()[0]
            if doc_num in ranker_scores[qid]:
                ranker_scores[qid][doc_num]['BM25'] = float(line.strip().split()[1])
            else:
                ranker_scores[qid][doc_num] = {}
                ranker_scores[qid][doc_num]['BM25'] = float(line.strip().split()[1])
            if float(line.strip().split()[1]) < min_bm25_score:
                min_bm25_score = float(line.strip().split()[1])
    with open('../line_predictions/'+ str(qid+1) + '.txt' ,'r') as infile:
        for line in infile:
            doc_num = line.strip().split()[0]
            if doc_num in ranker_scores[qid]:
                ranker_scores[qid][doc_num]['MeSH'] = float(line.strip().split('\t')[1])
            else:
                ranker_scores[qid][doc_num] = {}
                ranker_scores[qid][doc_num]['MeSH'] = float(line.strip().split('\t')[1])
            if float(line.strip().split()[1]) < min_mesh_score:
                min_mesh_score = float(line.strip().split()[1])
print ('Min mesh scores', min_mesh_score)
print ('Min bm25 scores', min_bm25_score)
for qid in range(50):
    print (qid)    
    bm25_scores = []
    mesh_scores =[]
    doc_numbers_1 = []
    doc_numbers_2 = []
    for doc_num in ranker_scores[qid]:
        if 'BM25' in ranker_scores[qid][doc_num]:
            bm25_scores += [ranker_scores[qid][doc_num]['BM25']]
            doc_numbers_1 += [doc_num]
        if 'MeSH' in ranker_scores[qid][doc_num]:
            mesh_scores += [ranker_scores[qid][doc_num]['MeSH']]
            doc_numbers_2 += [doc_num]
    scaler = MinMaxScaler(feature_range=(0.0001,1))
    print (qid)
    #print (doc_numbers_1)
    #print (doc_numbers_2)
    bm25_scores = scaler.fit_transform(np.reshape(np.array(bm25_scores), (-1,1))).reshape(1,-1).tolist()[0]
    mesh_scores = scaler.fit_transform(np.reshape(np.array(mesh_scores), (-1,1))).reshape(1,-1).tolist()[0]
    for idx,doc_num in enumerate(doc_numbers_1):
        ranker_scores[qid][doc_num]['BM25'] = bm25_scores[idx]
    for idx,doc_num in enumerate(doc_numbers_2):
        ranker_scores[qid][doc_num]['MeSH'] = mesh_scores[idx]
    for doc_num in ranker_scores[qid]:
        if 'BM25' not in ranker_scores[qid][doc_num]:
            ranker_scores[qid][doc_num]['BM25'] = 0
        if 'MeSH' not in ranker_scores[qid][doc_num]:
            ranker_scores[qid][doc_num]['MeSH'] = 0

    for idx,doc_num in enumerate(doc_numbers_1[:10]):
        print (bm25_scores[:10])
    for idx,doc_num in enumerate(doc_numbers_2[:10]):
        print (mesh_scores[:10])

best_factors1 = {}
best_factors2 = {} 
best_scores = {}
s = [np.linspace(0.0, 1.0, 6)]*2
for qid in range(50):
    rel_jud = {}
    best_factor1 = None
    best_factor2 = None
    best_score = 0
    best_ranked_docs = None
    with open('../jinfeng_data/data/jinfeng/true_positives/'+ str(qid+1) + '.txt' ,'r') as infile:
        for line in infile:
            rel_jud[line.strip()] = 1
    for factor1 in s[0]:
        for factor2 in s[1]:
            ranked_docs = {}
            for doc_num in ranker_scores[qid]:
                ranked_docs[doc_num] = factor1*ranker_scores[qid][doc_num]['BM25'] + factor2*ranker_scores[qid][doc_num]['MeSH']
            ranked_docs = sorted(ranked_docs.items(), key = lambda l :l[1], reverse = True)
            #print ranked_docs[:10]
            query_measures = evaluate_res(ranked_docs, rel_jud)
            if query_measures['ndcg10'] > best_score:
                best_score = query_measures['ndcg10']
                best_factor1 = factor1
                best_factor2 = factor2
                best_ranked_docs = ranked_docs
    print (qid)
    best_factors1[qid] = best_factor1
    best_factors2[qid] = best_factor2
    best_scores[qid] = best_score
    with open('../BM25_results/combined_predictions/'+ str(qid+1) + '.txt' ,'w') as outfile5:
        for doc in best_ranked_docs:
            outfile5.write(doc[0] + ' ' + str(doc[1]) + '\n')
for qid in range(50):
    print (qid,best_scores[qid],best_factors1[qid],best_factors2[qid])


print ('Average scores:', np.mean(best_scores.values()))
