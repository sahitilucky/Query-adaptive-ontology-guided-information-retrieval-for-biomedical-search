__author__ = 'Nick Hirakawa'
import sys
sys.path.append('../BM25-master/src/')

from Utils import *
from parse import *
from query import QueryProcessor
        
def main():
    #qp = QueryParser(filename='../BM25-master/text/Trec_2004_full_queries.txt')     #queries file
    #actual_queries = []
    #with io.open('../BM25-master/text/Trec_2004_full_queries.txt', 'r', encoding='utf-8') as Clickdata_list:
    #    for line in Clickdata_list:
    #        actual_queries += [line.strip()]
    cp = CorpusParser(filename='../BM25-master/text/trec_sample_abstract_corpus.txt')    #the product documents file
    i = 0 
    queries = []
    query_ids = []
    with io.open('../BM25-master/text/trec_sample_queries.txt', 'r', encoding='utf-8') as Clickdata_list:
        for line in Clickdata_list:
            if (i%2 ==0):
                query_ids += [line.strip().split('# ')[1]]
            else:
                queries += [line.strip().split()]
            i += 1
    #qp.parse()
    #queries = qp.get_queries()
    cp.parse()
    corpus = cp.get_corpus()
    print (len(corpus.keys()))
    print len(queries)
    proc = QueryProcessor(queries[9000:], corpus)
    
    print ('running BM25')
    results = proc.run()
    print ('queries done')
    qid = 0
    ranked_docs = []
    for result in results:
        sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1), reverse =True)
        ranked_docs += [sorted_x]
        qid += 1
    print (len(ranked_docs))
    Output_directory = '../BM25_results/' 
    with open('../BM25_results/bm25_mrrs_trec_titles.txt', 'a') as outfile:
	    for idx,query_id in enumerate(query_ids[9000:]):
	        with open('../BM25_results/trec_title_query_predictions/'+ str(query_id) + '.txt' ,'w') as outfile5:
	            for d_idx,doc in enumerate(ranked_docs[idx]):
	                outfile5.write(doc[0] + ' ' + str(doc[1]) + '\n')
	                if doc[0] == query_id:
	                    mrr = float(1)/float(d_idx+1)
	                    outfile.write(query_id + ' ' + str(mrr) + '\n')
	     
    #pickle.dump(ranked_docs, open(Output_directory + 'BM25_Trec_2004_query_results_full_corpus.p', 'w'))
    
    #ranked_docs = pickle.load(open(Output_directory + 'BM25_Trec_2004_query_results_full_corpus.p', 'r'))
#    with open(Output_directory + 'BM25_Trec_2004_query_results.json', 'w') as outfile:
#        json.dump(ranked_docs, outfile)
    '''
    print len(ranked_docs)
    measures = ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "ndcg100", "p5", "p10", "p1", "p3", "p100", "num_rel"]
    
    with open(Output_directory + 'BM25_Trec_2004_all_measures_full_query_12575.txt', 'w') as outfile:
        outfile.write(','.join([str(m) for m in measures]) + '\n')
        with open(Output_directory + 'BM25_Trec_2004_P10_full_query_12575.text', 'w') as outfile1:    
            with open(Output_directory + 'BM25_Trec_2004_NDCG10_full_query_12575.txt', 'w') as outfile2:    
                with open(Output_directory + 'BM25_Trec_2004_P100_full_query_12575.txt', 'w') as outfile3:    
                    with open(Output_directory + 'BM25_Trec_2004_NDCG100_full_query_12575.txt', 'w') as outfile4:    
                        for qid in range(50):
                            rel_jud = {}
                            with open('../jinfeng_data/data/jinfeng/true_positives/'+ str(qid+1) + '.txt' ,'r') as infile:
                            #with open('../true_positives/'+ str(qid+1) + '.txt' ,'r') as infile:
                                for line in infile:
                                    rel_jud[line.strip()] = 1
                            
                            with open('../BM25_results/full_query_predictions/'+ str(qid+1) + '.txt' ,'w') as outfile5:
                                for doc in ranked_docs[qid]:
                                    outfile5.write(doc[0] + ' ' + str(doc[1]) + '\n')
                            query_measures = evaluate_res(ranked_docs[qid], rel_jud)
                            outfile.write(','.join([str(query_measures[m]) for m in measures]) +'\n')
                            outfile1.write(str(query_measures['p10']) + '\n')
                            outfile2.write(str(query_measures['ndcg10']) + '\n')
                            outfile3.write(str(query_measures['p100']) + '\n')
                            outfile4.write(str(query_measures['ndcg100']) + '\n')
    '''
if __name__ == '__main__':
    main()