from expansion_model import ExpansionModel
from Utils import *
if __name__ == '__main__':
    
    '''
    Model init:
        cut_off: number of retrieved documents
        mesh_expansion_factor: number of nearest entities to use in mesh
        disgenet_expansion_factor: number of nearest entities to use in disgenet
        ontology_factor_mesh: ontology factor for mesh
    Output:
        ndcg_original: ndcg of original queries for cut_off
        ndcg_expanded: ndcg for expanded queries for cut_off
    '''
    
    em = ExpansionModel(cut_off = 20,
                        threshold = 0.5
                        )
    '''
    num_queries = 50
    mesh_sorted_res_all = []
    disgenet_sorted_res_all = []
    all_rel_jud = []
    text_bm25_results_file = 'data/results_newef_less_train/bm25_all_scores.txt'
    text_bm25_prediction = em.get_text_bm25_scores()[:num_queries]
    for i in range(1,num_queries+1):
        query_bm25_score_mesh_sorted, query_bm25_score_disgenet_sorted  = em.query_predict(query_id = str(i))
        rel_jud = {}
        for tp in em.true_positives[i]:
            rel_jud[tp] = 1
        
        mesh_sorted_res_all.append(query_bm25_score_mesh_sorted)
        disgenet_sorted_res_all.append(query_bm25_score_disgenet_sorted)
        all_rel_jud.append(rel_jud)
    write_scores(mesh_sorted_res_all,all_rel_jud,'data/results_newef_less_train/mesh_bm25_scores.txt')
    write_scores(disgenet_sorted_res_all,all_rel_jud,'data/results_newef_less_train/disgenet_bm25_scores.txt')
    write_scores(text_bm25_prediction,all_rel_jud,text_bm25_results_file)

    final_score_results_file = 'data/results_newef_less_train/combined_scores.txt'
    query_ids = [str(i) for i in range(1,num_queries+1)]
    ontology_factors = em.get_ontology_factors_algo1(query_ids)
    final_score = []
    for qid in range(num_queries):
        print (qid)
        final_score += [em.combine_scores(ontology_factors[str(qid+1)], [mesh_sorted_res_all[qid], disgenet_sorted_res_all[qid], text_bm25_prediction[qid]])]
    write_scores(final_score,all_rel_jud,final_score_results_file)
    '''
    '''
    bm25_scores_ndcg10 = []
    with open('data/results_withef/bm25_all_scores.txt' , 'r') as infile:
        for line in infile:
            bm25_scores_ndcg10 += [line.strip().split(',')[8]]
    mesh_scores_ndcg10 = []        
    with open('data/results_withef/mesh_bm25_scores.txt' , 'r') as infile:
        for line in infile:
            mesh_scores_ndcg10 += [line.strip().split(',')[8]]

    disgenet_scores_ndcg10 = []        
    with open('data/results_withef/disgenet_bm25_scores.txt' , 'r') as infile:
        for line in infile:
            disgenet_scores_ndcg10 += [line.strip().split(',')[8]]

    with open('data/analysis/only_ndcg10s.txt', 'w') as outfile:
        for i in range(len(bm25_scores_ndcg10)):
            outfile.write(mesh_scores_ndcg10[i] + ',' + disgenet_scores_ndcg10[i] + ',' + bm25_scores_ndcg10[i] + '\n')

    mesh_scores_ndcg10_before = []        
    with open('data/results/mesh_bm25_scores.txt' , 'r') as infile:
        for line in infile:
            mesh_scores_ndcg10_before += [line.strip().split(',')[8]]

    disgenet_scores_ndcg10_before = []        
    with open('data/results/disgenet_bm25_scores.txt' , 'r') as infile:
        for line in infile:
            disgenet_scores_ndcg10_before += [line.strip().split(',')[8]]


    with open('data/analysis/mesh_ef_compare.txt', 'w') as outfile1:
        with open('data/analysis/disgenet_ef_compare.txt', 'w') as outfile2:
            for i in range(1,len(mesh_scores_ndcg10)):
                after = float(mesh_scores_ndcg10[i])
                before = float(mesh_scores_ndcg10_before[i])
                if after >= before:
                    outfile1.write(mesh_scores_ndcg10[i] + ',' + mesh_scores_ndcg10_before[i] + '\t\t' + '1' + '\n')
                else:
                    outfile1.write(mesh_scores_ndcg10[i] + ',' + mesh_scores_ndcg10_before[i] + '\t\t' + '0' + '\n')
                after = float(disgenet_scores_ndcg10[i])
                before = float(disgenet_scores_ndcg10_before[i])
                if after >= before:
                    outfile2.write(disgenet_scores_ndcg10[i] + ',' + disgenet_scores_ndcg10_before[i] + '\t\t' + '1' + '\n')
                else:
                    outfile2.write(disgenet_scores_ndcg10[i] + ',' + disgenet_scores_ndcg10_before[i] + '\t\t' + '0' + '\n')
    '''
    '''
    mesh_prob_file = 'data/analysis/mesh_ep_biomed_dl_more_train.npy'
    mesh_mention_order_file = 'data/query_mesh_testing_data_order.txt'
    outfile = open('data/analysis/mesh_expansion_factors_biomed_dl.txt' , 'w')
    mesh_probs = np.load(mesh_prob_file)[:,1]
    mesh_mention_orders = []
    with open(mesh_mention_order_file,'r') as f:
        mesh_mention_orders = f.readlines()
    for i,mention in enumerate(mesh_mention_orders):
        qid = mention.split('\t')[0]
        outfile.write(mention.strip() + '\t' + str(mesh_probs[i])  +'\n')
    
    outfile.flush()
    outfile.close()

    disgenet_prob_file = 'data/analysis/disgenet_ep_biomed_dl_more_train.npy'
    disgenet_mention_order_file = 'data/disgenet_query_testing_data_order.txt'
    outfile = open('data/analysis/disgenet_expansion_factors_biomed_dl.txt' , 'w')
    disgenet_probs = np.load(disgenet_prob_file)[:,1]
    disgenet_mention_orders = []
    with open(disgenet_mention_order_file,'r') as f:
        disgenet_mention_orders = f.readlines()
    for i,mention in enumerate(disgenet_mention_orders):
        qid = mention.split('\t')[0]
        outfile.write(mention.strip() + '\t' + str(disgenet_probs[i])  +'\n')
    outfile.flush()
    outfile.close()
    '''
    
    for fold in range(1,6):
        ranked_docs = {}
        with open('data/OF_model_predictions_new_fold' + str(fold) + '.json', 'r') as infile:
            ranked_docs = json.load(infile)
        testing_queries = ranked_docs.keys()
        print (testing_queries)
        num_queries = 50
        text_bm25_prediction = em.get_text_bm25_scores()[:num_queries]
        all_rel_jud = []
        for i in range(1,num_queries+1):
            #query_bm25_score_mesh_sorted, query_bm25_score_disgenet_sorted  = em.query_predict(query_id = str(i))
            rel_jud = {}
            for tp in em.true_positives[i]:
                rel_jud[tp] = 1
            
            all_rel_jud.append(rel_jud)
        training_queries =[]
        
        qing_model_results = []
        test_all_rel_jud = []
        new_text_bm25_prediction = []
        for qid in ranked_docs:
            scores = [(x[0],float(x[1])) for x in ranked_docs[qid]]
            scores = sorted(scores,key = lambda l : l[1], reverse=True)
            qing_model_results += [scores]
            test_all_rel_jud += [all_rel_jud[int(qid)-1]]
            new_text_bm25_prediction += [text_bm25_prediction[int(qid)-1]]
        final_score_results_file = 'data/results_newef/OF_model_new_fold' + str(fold) + '_results.txt'
        write_scores(qing_model_results,test_all_rel_jud,final_score_results_file)
        final_score_results_file = 'data/results_newef/OF_model_new_fold' + str(fold) + '_bm_25_compare_results.txt'
        write_scores(new_text_bm25_prediction,test_all_rel_jud,final_score_results_file)
    for fold in range(1,6):
        ranked_docs = {}
        with open('data/Qing_glove_biomed_model_predictions_new_fold' + str(fold) + '.json', 'r') as infile:
            ranked_docs = json.load(infile)
        testing_queries = ranked_docs.keys()
        print (testing_queries)
        num_queries = 50
        text_bm25_prediction = em.get_text_bm25_scores()[:num_queries]
        all_rel_jud = []
        for i in range(1,num_queries+1):
            #query_bm25_score_mesh_sorted, query_bm25_score_disgenet_sorted  = em.query_predict(query_id = str(i))
            rel_jud = {}
            for tp in em.true_positives[i]:
                rel_jud[tp] = 1
            
            all_rel_jud.append(rel_jud)
        training_queries =[]
        
        qing_model_results = []
        test_all_rel_jud = []
        new_text_bm25_prediction = []
        for qid in ranked_docs:
            scores = [(x[0],float(x[1])) for x in ranked_docs[qid]]
            scores = sorted(scores,key = lambda l : l[1], reverse=True)
            qing_model_results += [scores]
            test_all_rel_jud += [all_rel_jud[int(qid)-1]]
            new_text_bm25_prediction += [text_bm25_prediction[int(qid)-1]]
        final_score_results_file = 'data/results_newef/Qing_glove_biomed_model_new_fold' + str(fold) + '_results.txt'
        write_scores(qing_model_results,test_all_rel_jud,final_score_results_file)
        final_score_results_file = 'data/results_newef/Qing_glove_biomed_model_new_fold' + str(fold) + '_bm_25_compare_results.txt'
        write_scores(new_text_bm25_prediction,test_all_rel_jud,final_score_results_file)
    