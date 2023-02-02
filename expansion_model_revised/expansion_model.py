import re
import math
import json
import os
import operator
import numpy as np
import pandas as pd
import random
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
#from gensim.summarization.bm25 import BM25
from scipy.spatial.distance import cosine
import sys
sys.path.append('../BM25-master/src/')
from parse import *
from query import QueryProcessor
from sklearn.preprocessing import MinMaxScaler
stopwords = stopwords.words()
lemmatizer = WordNetLemmatizer()

class ExpansionModel():
    # initialize factors from input
    def __init__(self,  
                 threshold = 0.1, cut_off = 10):
        
        #self.mesh_expansion_factor_dict = mesh_expansion_factor_dict
        #self.disgenet_expansion_factor_dict = disgenet_expansion_factor_dict
        #self.weight_mesh = ontology_factor_mesh
        #xself.weight_disgenet = 1 - ontology_factor_mesh
        self.cut_off = cut_off
        self.init_from_doc()
        self.threshold = threshold
        print("Finished loading data")

    def init_from_doc(self):
        '''
        corpus = {}
        with open('data/corpus.json') as corpus_file:
            for line in corpus_file:
                data = json.loads(line)
                pmid = data['pmid']
                abstract = data['abstract']
                if len(abstract) > 0:
                    corpus[int(pmid)] = abstract
        '''
        self.raw_queries = []
        with open('data/queries.txt') as f:
            for line in f:
                l = line.strip().split(', ')
                self.raw_queries.append(l)

        self.true_positives = {}
        for i in range(1, 51):
            tp_file = 'data/true_positives/%i.txt' % i
            pmid_list = []
            with open(tp_file) as f:
                for line in f:
                    pmid_list.append(line.strip())
            self.true_positives[i] = pmid_list
        '''
        mesh_queries_file = 'data/queries_mesh_manual_first50.txt'
        self.mesh_queries = []
        with open(mesh_queries_file) as f:
            for line in f:
                query = line.strip().split(', ')
                self.mesh_queries.append([x for x in query if x != 'NA'])
        '''
        mesh_id_dict_file = 'data/mh2id.txt'
        self.mesh_id_dict = {}
        with open(mesh_id_dict_file) as f:
            for line in f:
                l = line.strip().split('\t')
                self.mesh_id_dict[l[1]] = l[0]
        '''
        disgenet_linking_file = 'data/disgenet_link.txt'
        self.disgenet_linking = {}
        with open(disgenet_linking_file) as f:
            for line in f:
                l = line.strip().split(':')
                if len(l) == 1:
                    self.disgenet_linking[l[0]] = []
                else:
                    self.disgenet_linking[l[0]] = l[1].split(',')
        '''
        mesh_path_file = 'data/midpath_mh.txt'
        self.mesh_paths = []
        with open(mesh_path_file) as f:
            for line in f:
                mesh_path = line.strip().split('::')
                self.mesh_paths.append(mesh_path[2:])
        '''
        self.disgenet_queries = []
        with open('data/disgenet_query_link.txt') as f:
            for line in f:
                self.disgenet_queries.append(line.strip().split(', '))
        '''
        self.gene_disease_network = pd.read_csv('data/gene_disease_network.csv').drop('Unnamed: 0', axis = 1)
        disgenet_file = pd.read_csv('data/curated_gene_disease_associations.tsv', sep = '\t')

        disease_info = disgenet_file[['diseaseId', 'diseaseName']].drop_duplicates()
        self.disease_dict = {disgenet_file['diseaseId'].loc[x]: disgenet_file['diseaseName'].loc[x].lower() for x in disease_info.index}

        gene_info = disgenet_file[['geneId', 'geneSymbol']].drop_duplicates()
        self.gene_dict = {str(disgenet_file['geneId'].loc[x]): disgenet_file['geneSymbol'].loc[x] for x in gene_info.index}

        self.disgenet_id_dict = self.disease_dict.copy()
        self.disgenet_id_dict.update(self.gene_dict)
        print (self.disease_dict.keys()[:100])
        print (self.gene_dict.keys()[:100])

        gene_to_disease = {}
        disease_to_gene = {}
        i = 0
        with open('data/gene_disease_network.csv', 'r') as infile:
            for line in infile:
                if (i==0):
                    i =1
                    continue
                gene = line.split(',')[1]
                disease = line.split(',')[2]
                score = line.strip().split(',')[3]
                if float(score)<=0.03:
                    continue
                try:
                    gene_to_disease[gene] += [(disease,score)]
                except:
                    gene_to_disease[gene] = [(disease,score)]
                try:
                    disease_to_gene[disease] += [(gene,score)]
                except:
                    disease_to_gene[disease] = [(gene,score)]
        self.disease_to_gene = disease_to_gene
        self.gene_to_disease = gene_to_disease
        '''
        self.pmid_order = []
        self.bm25_docs = []
        with open('data/bm25_docs.txt') as f:
            for line in f:
                l = line.strip().split(':')
                self.pmid_order.append(int(l[0]))
                self.bm25_docs.append(l[1].split(';'))
                
        self.bm25_doc_dict = {self.pmid_order[i]: self.bm25_docs[i] for i in range(len(self.pmid_order))}
        '''
        #ID DOCS
        #disgenet
        disgenet_corpus_id_docs = {}
        with open('data/doc_disgenet_entities.txt', 'r') as infile:
            for line in infile:
                if len(line.strip().split('\t'))==2:
                    [doc_id,mesh_ids] = line.strip().split('\t')
                    disgenet_corpus_id_docs[doc_id] = mesh_ids.split(';')
                else:
                    [doc_id] = line.strip().split('\t')
                    disgenet_corpus_id_docs[doc_id] = []
        self.disgenet_corpus_id_docs = disgenet_corpus_id_docs
        #MESH
        #entity 'C' to mesh 'D'
        entity_to_mesh_links = {}
        with open('data/entity_mh_links.json', 'r') as infile:
            entity_to_mesh_links = json.load(infile) 

        corpus_id_docs = {}
        with open('data/doc_pubtator_entities.txt', 'r') as infile:
            for line in infile:
                if len(line.strip().split('\t'))==2:
                    [doc_id,mesh_ids] = line.strip().split('\t')
                    corpus_id_docs[doc_id] = mesh_ids.split(';')
                else:
                    [doc_id] = line.strip().split('\t')
                    corpus_id_docs[doc_id] = []
        with open('data/doc_pubmed_mh.txt', 'r') as infile:
            for line in infile:
                if len(line.strip().split('\t'))==2:
                    [doc_id,mesh_ids] = line.strip().split('\t')
                    pubmed_mesh_ids = mesh_ids.split(';')
                else:
                    [doc_id] = line.strip().split('\t')
                    pubmed_mesh_ids = []
                corpus_doc_id_dict = dict(zip(corpus_id_docs[doc_id], range(len(corpus_id_docs[doc_id]))))
                for mesh_id in pubmed_mesh_ids:
                    if mesh_id not in corpus_doc_id_dict:
                        corpus_id_docs[doc_id] += [mesh_id]
        for doc_id in corpus_id_docs:
            mesh_id_list = corpus_id_docs[doc_id]
            corpus_id_docs[doc_id] = []
            for mesh_id in mesh_id_list:
                if mesh_id[0] == 'C':
                    if mesh_id in entity_to_mesh_links:
                        corpus_id_docs[doc_id] += entity_to_mesh_links[mesh_id]
                    else:
                        corpus_id_docs[doc_id] += [mesh_id] 
                else:
                    corpus_id_docs[doc_id] += [mesh_id]
        self.corpus_id_docs = corpus_id_docs
        self.proc = QueryProcessor([], self.corpus_id_docs)
        self.disgenet_proc = QueryProcessor([], self.disgenet_corpus_id_docs)
        #mesh queries
        mesh_queries = {}
        disgenet_queries = {}
        with open('data/queries_mention.txt', 'r') as infile:
            for line in infile:
                line_list = line.strip().split('\t')
                query_id = line_list[0]
                if line_list[-1][:5] == 'MESH:':
                    if query_id in mesh_queries:
                        mesh_queries[query_id] += [line_list[-1]] 
                    else:
                        mesh_queries[query_id] = [line_list[-1]]
                    entityid = line_list[-1] 
                    entityid = entityid[5:]
                    if (entityid[0] == 'C'):
                        if entityid in entity_to_mesh_links:
                            mesh_queries[query_id][-1] = entity_to_mesh_links[entityid] 
                        else:
                            #print ('gone')
                            mesh_queries[query_id][-1] = [entityid]
                    else:
                        mesh_queries[query_id][-1] = [entityid] 
                else:
                    if query_id in disgenet_queries:
                        disgenet_queries[query_id] += [line_list[-1]] 
                    else:
                        disgenet_queries[query_id] = [line_list[-1]]
        for query_id in mesh_queries:
            if query_id not in disgenet_queries:
                disgenet_queries[query_id] = []
        self.mesh_queries = mesh_queries
        self.disgenet_queries = disgenet_queries

        #disgenet queries

        mesh_embedding_file = 'data/mesh_undirected_embedding_concat.csv'
        mesh_embedding = {}
        with open(mesh_embedding_file) as f:
            for line in f:
                l = line.strip().split(' ')
                if len(l) > 2:
                    mesh_embedding[l[0]] = np.array([float(x) for x in l[1:]])
        
        self.mesh_ids = list(mesh_embedding.keys())
        self.mesh_emb_matrix = np.array(list(mesh_embedding.values()))
        
        disgenet_embedding_file = 'data/disgenet_undirected_embedding_concat.csv'
        disgenet_embedding = {}
        with open(disgenet_embedding_file) as f:
            for line in f:
                l = line.strip().split(' ')
                if len(l) > 2:
                    disgenet_embedding[l[0]] = np.array([float(x) for x in l[1:]])
                
        self.disgenet_ids = list(disgenet_embedding.keys())
        self.disgenet_emb_matrix = np.array(list(disgenet_embedding.values()))


        mesh_prob_file = 'data/mesh_ep_glove_nf_dl.npy'
        mesh_mention_order_file = 'data/query_mesh_testing_data_order.txt'
        outfile = open('data/analysis/mesh_expansion_factors_mlp.txt' , 'w')
        mesh_probs = np.load(mesh_prob_file)[:,1]
        self.mesh_expansion_factor_list = {}
        with open(mesh_mention_order_file,'r') as f:
            mesh_mention_orders = f.readlines()
        for i,mention in enumerate(mesh_mention_orders):
            qid = mention.split('\t')[0]
            outfile.write(qid + ' ' + str(mesh_probs[i])  +'\n')
            try:
                self.mesh_expansion_factor_list[qid].append(mesh_probs[i])
            except:
                self.mesh_expansion_factor_list[qid] = [mesh_probs[i]]

        for qid,query in self.mesh_queries.items():
            print (qid,query)
            print (self.mesh_expansion_factor_list[qid])
        outfile.flush()
        outfile.close()

        disgenet_prob_file = 'data/disgenet_ep_biomed_dl_more_train.npy'
        disgenet_mention_order_file = 'data/disgenet_query_testing_data_order.txt'
        outfile = open('data/analysis/disgenet_expansion_factors_mlp.txt' , 'w')
        disgenet_probs = np.load(disgenet_prob_file)[:,1]
        self.disgenet_expansion_factor_list = {}
        with open(disgenet_mention_order_file,'r') as f:
            disgenet_mention_orders = f.readlines()
        for i,mention in enumerate(disgenet_mention_orders):
            qid = mention.split('\t')[0]
            outfile.write(qid + ' ' + str(disgenet_probs[i])  +'\n')
            try:
                self.disgenet_expansion_factor_list[qid].append(disgenet_probs[i])
            except:
                self.disgenet_expansion_factor_list[qid] = [disgenet_probs[i]]
        for qid in self.mesh_expansion_factor_list:
            if qid not in self.disgenet_expansion_factor_list:
                self.disgenet_expansion_factor_list[qid] = []
        outfile.flush()
        outfile.close()
        #initialization
        #mesh_expansion_factor_list = {}
        #disgenet_expansion_factor_list = {}
        #for query_id in self.mesh_queries:
        #    mesh_expansion_factor_list[query_id] = [1]*len(self.mesh_queries[query_id])
        #
        #for query_id in self.disgenet_queries:
        #    disgenet_expansion_factor_list[query_id] = [1]*len(self.disgenet_queries[query_id])
        #self.mesh_expansion_factor_list = mesh_expansion_factor_list
        #self.disgenet_expansion_factor_list = disgenet_expansion_factor_list


        #sample = random.sample(self.corpus_id_docs.keys(), 10)
        #for s in sample:
        #    print (s)
        #    print (self.corpus_id_docs[s])
    def mesh_id_to_term(self, candidate):
        return {self.mesh_id_dict[t]: candidate[t] for t in candidate.keys() 
                if t in self.mesh_id_dict.keys()}
    
    def disgenet_id_to_term(self, candidate):
        return {self.disgenet_id_dict[t]: candidate[t] for t in candidate.keys() 
                if t in self.disgenet_id_dict.keys()}
        
    def preprocess(self, text):
        text = re.sub(r'[^\w\s]','',text)
        words = word_tokenize(text.lower())
        lemmas = []
        for word in words:
            if word not in stopwords:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemmas.append(lemma)

        text = ' '.join(lemmas)
        return text

    def get_ndcg(self, sorted_res, rel_jud, cutoff):
        dcg = 0
        for i in range(min(cutoff, len(sorted_res))):
            doc_id = sorted_res[i]
            if doc_id in rel_jud:
                rel_level = 1
            else:
                rel_level = 0
            dcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))

        ideal_sorted = {}
        for tup in sorted_res:
            try:
                if tup in rel_jud:
                    ideal_sorted[tup] = 1
                else:
                    ideal_sorted[tup] = 0
            except:
                ideal_sorted[tup] = 0
        ideal_sorted = sorted(ideal_sorted.items(), key=operator.itemgetter(1), reverse=True)
        ideal_sorted = [x[0] for x in ideal_sorted]

        idcg = 0
        for i in range(min(cutoff, len(ideal_sorted))):
            doc_id = ideal_sorted[i]
            if doc_id in rel_jud:
                rel_level = 1
            else:
                rel_level = 0
            idcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))

        if idcg == 0:
            idcg = 1

        return dcg/idcg

    def get_mesh_candidates(self, term):
        candidates = []
        for path in self.mesh_paths:
            if term in path:
                candidates += path[path.index(term):][:5]
        return list(set(candidates))
    
    def get_disgenet_candidates(self, term):
        term_expan = []
        if term in self.disease_dict.keys():
            genes_df = self.gene_disease_network.where(self.gene_disease_network['diseaseId'] == term).dropna()
            #sorted_genes = sorted(list(set([(genes_df['score'].loc[x], genes_df['geneId'].loc[x]) 
            #                                for x in genes_df.index])), reverse = True)
            #genes = [int(x[1]) for x in sorted_genes]
            genes = list(genes_df['geneId'].unique())
            
            term_expan = [str(int(x)) for x in genes]
            for gene_id in genes:
                disease_df = self.gene_disease_network.where(self.gene_disease_network['geneId'] == int(gene_id)).dropna()
                sorted_diseases = sorted(list(set([(disease_df['score'].loc[x], disease_df['diseaseId'].loc[x]) 
                                              for x in disease_df.index])), reverse = True)
                diseases = [x[1] for x in sorted_diseases][:5]
                term_expan += diseases

        elif term in self.gene_dict.keys():
            disease_df = self.gene_disease_network.where(self.gene_disease_network['geneId'] == int(term)).dropna()
            #sorted_diseases = sorted(list(set([(disease_df['score'].loc[x], disease_df['diseaseId'].loc[x]) 
            #                                  for x in disease_df.index])), reverse = True)
            #diseases = [x[1] for x in sorted_diseases]
            diseases = list(disease_df['diseaseId'].unique())
            term_expan = diseases[:]
            for disease_id in diseases:
                genes_df = self.gene_disease_network.where(self.gene_disease_network['diseaseId'] == disease_id).dropna()
                sorted_genes = sorted(list(set([(genes_df['score'].loc[x], genes_df['geneId'].loc[x]) 
                                            for x in genes_df.index])), reverse = True)
                genes = [str(int(x[1])) for x in sorted_genes][:5]
                term_expan += genes 
        
        return list(set(term_expan))

    def get_disgenet_candidates_2(self, term):
        term_expan = []
        if term in self.disease_to_gene:
            genes_df = self.disease_to_gene[term]
            #sorted_genes = sorted(list(set([(genes_df['score'].loc[x], genes_df['geneId'].loc[x]) 
            #                                for x in genes_df.index])), reverse = True)
            #genes = [int(x[1]) for x in sorted_genes]
            term_expan = [x[0] for x in genes_df]
            for gene in genes_df:
                if gene[0] in self.gene_to_disease:
                    scores = self.gene_to_disease[gene[0]]
                    top_5_diseases = sorted(scores, key = lambda l :l[1], reverse=True)[:5]
                    term_expan += [x[0] for x in top_5_diseases]
            
        elif term in self.gene_to_disease:
            diseases_df = self.gene_to_disease[term]
            #sorted_genes = sorted(list(set([(genes_df['score'].loc[x], genes_df['geneId'].loc[x]) 
            #                                for x in genes_df.index])), reverse = True)
            #genes = [int(x[1]) for x in sorted_genes]
            term_expan = [x[0] for x in diseases_df]
            for disease in diseases_df:
                if disease[0] in self.disease_to_gene:
                    scores = self.disease_to_gene[disease[0]]
                    top_5_genes = sorted(scores, key = lambda l :l[1], reverse=True)[:5]
                    term_expan += [x[0] for x in top_5_genes]
            
        return list(set(term_expan))
    
    
    def term_expand(self, term, candidates, ids, emb_matrix, expansion_factor):
        cand_ids = [ids.index(str(cand)) for cand in candidates]
        '''
        return: dictionary of candidate : expansion_factor*cosine similarity of valid candidates
        '''
        if term not in ids:
            return {term: 1}
        ind = ids.index(term)
        cosine_similarities = [abs(1-cosine(emb_matrix[ind], emb_matrix[j])) for j in cand_ids]
        ranked_ind = np.argsort(cosine_similarities)[::-1][1:]
        ranked_candidates = {}
        
        for j in ranked_ind:
            if expansion_factor*cosine_similarities[j] > self.threshold:
                ranked_candidates[candidates[j]] = expansion_factor*cosine_similarities[j]

        ranked_candidates = dict(sorted(ranked_candidates.items(), key =lambda l:l[1],reverse=True)[:10])
        return ranked_candidates
    
    def mesh_query_expand(self, query, expansion_factor_list):
        #query = [x for x in query if x in self.mesh_ids
        #query_expand_dict = {x: 1 for x in query}
        query_expand_dict = {}
        for idx,term_list in enumerate(query):
            for term in term_list:
                if term in query_expand_dict:
                    query_expand_dict[term] += 1 
                else:
                    query_expand_dict[term] = 1
                candidates = self.get_mesh_candidates(term)
                if term in self.mesh_ids:
                    ranked_candidates = self.term_expand(term, candidates, self.mesh_ids, 
                                                         self.mesh_emb_matrix, 
                                                         expansion_factor_list[idx])
                    print (ranked_candidates)
                    for candidate in ranked_candidates:
                        if candidate in query_expand_dict:
                            query_expand_dict[candidate] += ranked_candidates[candidate] 
                        else:
                            query_expand_dict[candidate] = ranked_candidates[candidate]
        return query_expand_dict
    
    def disgenet_query_expand(self, query, expansion_factor_list):
        #query = [x for x in query if x in self.disgenet_ids]
        query_expand_dict = {}
        for idx,term in enumerate(query):
            if term in self.disgenet_ids:
                if term in query_expand_dict:
                    query_expand_dict[term] += 1 
                else:
                    query_expand_dict[term] = 1
                candidates = self.get_disgenet_candidates_2(term)
                #print (candidates)
                ranked_candidates = self.term_expand(term, candidates, self.disgenet_ids, 
                                                     self.disgenet_emb_matrix, 
                                                     expansion_factor_list[idx])
                for candidate in ranked_candidates:
                    if candidate in query_expand_dict:
                        query_expand_dict[candidate] += ranked_candidates[candidate] 
                    else:
                        query_expand_dict[candidate] = ranked_candidates[candidate]
                #print (ranked_candidates)
                
        return query_expand_dict
    
    def query_predict(self, query_id):
        
        #print('Raw query: ', self.raw_queries[query_id])
        print (self.threshold)
        mesh_query = self.mesh_queries[query_id]
        print (query_id)
        #print('Expanding on MeSH ...')
        #print (mesh_query)
        mesh_expand_entity = self.mesh_query_expand(mesh_query,self.mesh_expansion_factor_list[query_id])
        #mesh_expand_term = self.mesh_id_to_term(mesh_expand_entity)
        #print('MeSH expansion: ', len(mesh_expand_entity.keys()))
        #print ('Mesh score:')
        mesh_bm25_dict = self.query_bm25(mesh_expand_entity, self.corpus_id_docs, self.proc)
        #print('Finished MeSH BM25!')
        mesh_bm25_scores_sorted = sorted(mesh_bm25_dict.iteritems(), key=operator.itemgetter(1), reverse =True)
        #print (mesh_bm25_scores_sorted[:10])    
        #print (self.true_positives[int(query_id)][:100])


        disgenet_query = self.disgenet_queries[query_id]
        #print('Expanding on DisGenet ...')
        #print (disgenet_query)
        disgenet_expand_entity = self.disgenet_query_expand(disgenet_query,self.disgenet_expansion_factor_list[query_id])
        #disgenet_expand_term = self.disgenet_id_to_term(disgenet_expand_entity)
        #print('DisGenet expansion: ', len(disgenet_expand_entity.keys()))
        #print ('Disgenet score:')  
        disgenet_bm25_dict = self.query_bm25(disgenet_expand_entity, self.disgenet_corpus_id_docs, self.disgenet_proc)
        #print('Finished DisGenet BM25!')
        disgenet_bm25_scores_sorted = sorted(disgenet_bm25_dict.iteritems(), key=operator.itemgetter(1), reverse =True)
        #print (disgenet_bm25_scores_sorted[:10])
        #print (em.true_positives[int(query_id)][:100])
        return mesh_bm25_scores_sorted, disgenet_bm25_scores_sorted
    
    
    def query_bm25(self, query_ranked_candidates, corpus_id_docs,proc):
        '''
        Input:
            ranked_candidates: key:candidate ID, output from term_expan
        Output:
            bm25_score_dict: key: pmid, value: bm25 score
        '''
        # Sahiti
        #print (query_ranked_candidates.items())
        bm25_score_dict = proc.run_with_weights(query_ranked_candidates.items())
        
        return bm25_score_dict

    def combine_scores(self,ontology_factors, different_scores_list):
        scaler = MinMaxScaler(feature_range=(0.0001,1))
        all_doc_ids = []
        number_of_sources = len(different_scores_list)
        normalized_scores_list = []
        print (ontology_factors)
        for scores_list in different_scores_list:
            scores_list_ids = [s[0] for s in scores_list]
            scores_list_values = [s[1] for s in scores_list]
            if scores_list_values == []:
                normalized_scores_list += [dict(scores_list)]
            else:
                scores_list_values = scaler.fit_transform(np.reshape(np.array(scores_list_values), (-1,1))).reshape(1,-1).tolist()[0]
                scores_list = zip(scores_list_ids, scores_list_values)
                #print scores_list[:10]
                normalized_scores_list += [dict(scores_list)]
            all_doc_ids += scores_list_ids      
        all_doc_ids = list(set(all_doc_ids))
        final_scores = {}
        for doc_id in all_doc_ids:
            final_scores[doc_id] = 0
            for i in range(number_of_sources):
                if doc_id in normalized_scores_list[i]:
                    final_scores[doc_id] = ontology_factors[i]*normalized_scores_list[i][doc_id]
        #print normalized_scores_list[0][:10]            
        final_scores = sorted(final_scores.items(), key=lambda l : l[1], reverse=True)
        #print final_scores[:10]
        return final_scores

    def get_ontology_factors_algo1(self,query_ids):
        ontology_factors = {}
        scaler = MinMaxScaler(feature_range=(0.0001,1))
        avg_mesh = []
        avg_disgenet = []
        disgenet_query_ids = []
        for query_id in query_ids:
            avg_mesh += [float(sum(self.mesh_expansion_factor_list[query_id]))/float(len(self.mesh_expansion_factor_list[query_id]))]
            if len(self.disgenet_expansion_factor_list[query_id]) > 0:
                avg_disgenet += [float(sum(self.disgenet_expansion_factor_list[query_id]))/float(len(self.disgenet_expansion_factor_list[query_id]))]
                disgenet_query_ids += [query_id]
        avg_mesh = scaler.fit_transform(np.reshape(np.array(avg_mesh), (-1,1))).reshape(1,-1).tolist()[0]
        avg_disgenet = scaler.fit_transform(np.reshape(np.array(avg_disgenet), (-1,1))).reshape(1,-1).tolist()[0]
        outfile = open('data/analysis/ontology_factors_and_features.txt', 'w')
        for query_id in query_ids:
            avg_mesh_ef = avg_mesh[int(query_id)-1]
            if query_id in disgenet_query_ids:
                avg_disgenet_ef = avg_disgenet[disgenet_query_ids.index(query_id)]
            else:
                avg_disgenet_ef = 0.0001
            mesh_query = self.mesh_queries[query_id]
            disgenet_query = self.disgenet_queries[query_id]   
            mesh_weight = float(len(mesh_query))/float(len(mesh_query) + len(disgenet_query))
            disgenet_weight = float(len(disgenet_query))/float(len(mesh_query) + len(disgenet_query))
            mesh_weight = mesh_weight*(math.exp(avg_mesh_ef)/float(1+math.exp(avg_mesh_ef)))
            disgenet_weight = disgenet_weight*(math.exp(avg_disgenet_ef)/float(1+math.exp(avg_disgenet_ef)))
            #bm25_weight = float(2)/float(2+avg_mesh_ef+avg_disgenet_ef)
            avg_expansion_factor = float(avg_mesh_ef+avg_disgenet_ef)/float(2)
            bm25_weight = float(2)/float(1+math.exp(avg_expansion_factor))
            ontology_factors[query_id] = [mesh_weight,disgenet_weight, bm25_weight]
            #ontology_factors[query_id] = [1, 1, 1]
            outfile.write(query_id + ',' + str(mesh_weight) + ',' + str(disgenet_weight) + ',' + str(bm25_weight) + ',' + str(len(mesh_query)) +',' +  str(len(disgenet_query)) +',' + str(avg_expansion_factor) + ',' + str(avg_mesh_ef) + ',' + str(avg_disgenet_ef) + '\n')
        outfile.flush()
        outfile.close()
        return ontology_factors
    def get_ontology_factors_algo2(query_ids):
        #using average embedding similarity
        return

    def get_text_bm25_scores(self):
        #true_pos_dir = '/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/true_positives'
        pred_dir= 'data/full_query_predictions'
        q_files = [str(i+1)+'.txt' for i in range(50)]
        all_sorted_res = []
        all_rel_jud = []
        for qf in q_files:
            '''
            with open(os.path.join(true_pos_dir,qf),'r') as f:
                rel_jud = {}
                tps = [tp.strip('\n') for tp in f.readlines()]
                for tp in tps:
                    rel_jud[tp] = 1
                all_rel_jud.append(rel_jud)
            '''
            with open(os.path.join(pred_dir,qf),'r') as f:
                sorted_res = []
                preds = f.readlines()
                for p in preds:
                    [doc,score] = p.split()
                    sorted_res.append(tuple((doc,float(score))))
                all_sorted_res.append(sorted_res)
        return all_sorted_res
    '''
    def bm25_ndcg(self, query_term, expand_query, raw_query, 
                  bm25_docs, pmid_order, query_true_positives, cut_off):

        bm25 = BM25(bm25_docs)
        score_original = bm25.get_scores(' '.join(raw_query).split(' '))
        score_expand = bm25.get_scores(' '.join(expand_query + raw_query).split(' ') + expand_query)
        sorted_ind_original = np.argsort(score_original)[::-1]
        sorted_ind_expand = np.argsort(score_expand)[::-1]

        retrieved_pmid_original = [pmid_order[i] for i in sorted_ind_original[:cut_off]]
        retrieved_pmid_expand = [pmid_order[i] for i in sorted_ind_expand[:cut_off]]

        ndcg_original = self.get_ndcg(retrieved_pmid_original, query_true_positives, cut_off)
        ndcg_expand = self.get_ndcg(retrieved_pmid_expand, query_true_positives, cut_off)

        return ndcg_original, ndcg_expand, retrieved_pmid_original, retrieved_pmid_expand

    def combined_result(self, mesh_queries, disgenet_queries, raw_queries, 
                        mesh_ids, mesh_emb_matrix, k_mesh, disgenet_ids, 
                        disgenet_emb_matrix, k_disgenet, bm25_docs, 
                        pmid_order, true_positives, disgenet_id_dict, mesh_id_dict, 
                        mesh_paths, gene_disease_network, disease_dict, gene_dict,
                        cut_off, weight_mesh, weight_disgenet):
        ndcg_o = []
        ndcg_c = []
        for i in range(50):
            print("Evaluating %ith query"%i)
            raw_query = raw_queries[i]
            disgenet_query = disgenet_queries[i]
            disgenet_query_expand = self.disgenet_query_expand(disgenet_query, disgenet_ids, 
                                                      disgenet_emb_matrix, k_disgenet,
                                                      gene_disease_network, disease_dict, gene_dict)
            disgenet_query_expand_term = self.id_to_term(disgenet_query_expand, 
                                                         disgenet_id_dict)

            mesh_query = disgenet_queries[i]
            mesh_query_expand = self.mesh_query_expand(mesh_query, mesh_ids, 
                                                       mesh_emb_matrix, k_mesh, mesh_paths)
            mesh_query_expand_term = self.id_to_term(mesh_query_expand, mesh_id_dict)


            bm25 = BM25(bm25_docs)
            score_original = bm25.get_scores(' '.join(raw_query).split(' '))
            score_mesh_expand = bm25.get_scores(' '.join(mesh_query_expand_term + raw_query).split(' ') + 
                                                mesh_query_expand_term)
            score_disgenet_expand = bm25.get_scores(' '.join(disgenet_query_expand_term + raw_query).split(' ') + 
                                                    disgenet_query_expand_term)
            combined_score = [weight_mesh*score_mesh_expand[j] + weight_disgenet*score_disgenet_expand[j] 
                              for j in range(len(score_original))]

            sorted_ind_original = np.argsort(score_original)[::-1]
            sorted_ind_combined = np.argsort(combined_score)[::-1]

            retrieved_pmid_original = [pmid_order[j] for j in sorted_ind_original[:cut_off]]
            retrieved_pmid_combined = [pmid_order[j] for j in sorted_ind_combined[:cut_off]]

            query_true_positives = true_positives[i+1]
            ndcg_original = self.get_ndcg(retrieved_pmid_original, query_true_positives, cut_off)
            ndcg_combined = self.get_ndcg(retrieved_pmid_combined, query_true_positives, cut_off)

            ndcg_o.append(ndcg_original)
            ndcg_c.append(ndcg_combined)

        return ndcg_o, ndcg_c

    def predict(self):
        ndcg_o, ndcg_e = self.combined_result(self.mesh_queries, self.disgenet_queries, 
                                              self.raw_queries, self.mesh_ids, self.mesh_emb_matrix, 
                                              self.MESH_FACTOR, self.disgenet_ids, self.disgenet_emb_matrix, 
                                              self.DISGENET_FACTOR, self.bm25_docs, self.pmid_order, 
                                              self.true_positives, self.disgenet_id_dict, self.mesh_id_dict, 
                                              self.mesh_paths, self.gene_disease_network, self.disease_dict, self.gene_dict,
                                              self.cut_off, self.WEIGHT_MESH, self.WEIGHT_DISGENET)
    
        return ndcg_o, ndcg_e
    '''
