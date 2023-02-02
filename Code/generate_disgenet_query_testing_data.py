from Utils import *
#TASKS
#1.Sample points accordingly - get document list with positives, get negative list as well, include get document list of those 12575 in sample
#Get word embeddings
#Generate training data for the sampled points
#loadinf input files
print ('loading....')
context_statistics = {}
with open('../LC_related_data_2/disgenet_context_statistics.txt', 'r') as infile:
	for line in infile:
		[word,pos_freq,total_freq]= line.strip().split('\t')
		context_statistics[word] = [float(pos_freq),float(total_freq)]

types_statistics ={}
with open('../LC_related_data_2/disgenet_types_statistics.txt', 'r') as infile:
	for line in infile:
		[word,pos_freq,total_freq]= line.strip().split('\t')
		types_statistics[word] = [float(pos_freq),float(total_freq)]

context_word_vocabulary = {}
with open('../LC_related_data_2/disgenet_context_word_vocabulary.json', 'r') as infile:
	context_word_vocabulary = json.load(infile)

positive_document_dict = {}
with open('../LC_related_data_2/disgenet_positive_document_dict.json', 'r') as infile:
	positive_document_dict = json.load(infile)
	
negative_document_dict = {}
with open('../LC_related_data_2/disgenet_negative_document_dict.json', 'r') as infile:
	negative_document_dict = json.load(infile)
'''
#get documents content
corpus = {}
with open('../data_files/raw_queries_2004.txt', 'r') as infile:
	i = 1
	for line in infile:
		corpus[str(i)] = line.strip()
		i += 1
'''
corpus1 = {}
with open('../data_files/bioconcepts2pubtator_trec_sample_title_abstracts_corpus.json', 'r') as outfile:
    corpus1 = json.load(outfile)
corpus = {}
for pmid in corpus1:
	corpus[pmid] = corpus1[pmid]['title']

context_word_idfs = {}
with open('../data_files/bioconcepts2pubtator_corpus_idfs.json', 'r') as infile:
    context_word_idfs = json.load(infile)
print ('loading done....')

types_index_dict = json.load(open('../LC_related_data_2/disgenet_types_index_dict.json','r'))
context_word_vocabulary_dict = json.load(open('../LC_related_data_2/disgenet_context_word_vocabulary_dict.json','r'))

#bioconcept input files - getting local context ad bigger type
doc_entities = {}
query_ids = []
query_ids_dict = {}
i = 0 
#inputfile = '../jinfeng_data/data/jinfeng/pubtator_mention_2_meshid.txt'
inputfile = '../data_files/bioconcepts2pubtator_mesh_disgent_title_mentions.txt'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num not in query_ids_dict:
			query_ids += [doc_num]
			query_ids_dict[doc_num] = 1
		if line_list[-1][:5] != 'MESH:':
			print (line)
			if doc_num in doc_entities:
				doc_entities[doc_num] += [line_list[1:]] 
			else:
				doc_entities[doc_num] = [line_list[1:]]
		i += 1
		if (i%100) == 0:
			print (i)
print (doc_entities.keys())
word_idf_list = context_word_idfs.values()
average_word_idf = float(sum(word_idf_list))/float(len(word_idf_list))
#create training data
outfile1 = open('../LC_related_data_2/disgenet_trec_title_query_testing_data.txt', 'w')
outfile2 = open('../LC_related_data_2/disgenet_trec_title_query_testing_data_order.txt', 'w')
i = 0
num_features_c = len(context_word_vocabulary_dict.keys())
num_features_t = len(types_index_dict.keys())
print (num_features_c,num_features_t)
for doc_num in query_ids:
	doc_num = str(doc_num)
	if doc_num in doc_entities:
		for entity in doc_entities[doc_num]:
			feature_c = [0]*(num_features_c)
			feature_c_2 = [0]*(num_features_c)
			feature_c_3 = [0]*(num_features_c)
			feature_t = [0]*(num_features_t)
			feature_t[types_index_dict[entity[3]]] = float(types_statistics[entity[3]][0])/float(types_statistics[entity[3]][1])
			left_context = corpus[doc_num][:int(entity[0])].split()[-5:]
			right_context = corpus[doc_num][int(entity[1]):].split()[:5]
			context_words = preprocess(' '.join(left_context)).split()
			context_words += preprocess(' '.join(right_context)).split()
			context_words_dict = dict(zip(context_words,range(len(context_words))))
			for word in context_words_dict:
				try:
					word_idf = context_word_idfs[word]
				except:
					word_idf = average_word_idf
				try:
					word_pos_freq = context_statistics[word][0]
					word_total_freq = context_statistics[word][1]
					word_prob = float(word_pos_freq)/float(word_total_freq)
				except:
					#print ('came here')
					word_prob = 0
				try: 
					feature_c[context_word_vocabulary_dict[word]] = word_prob
					feature_c_2[context_word_vocabulary_dict[word]] = word_idf
					feature_c_3[context_word_vocabulary_dict[word]] = 1
				except:
					pass
			feature_c.extend(feature_c_2)
			feature_c.extend(feature_c_3)
			feature_c.extend(feature_t)
			final_feature = feature_c
			final_feature_string = ','.join([str(s) for s in final_feature])
			outfile1.write(final_feature_string + '\n')
			outfile2.write(doc_num + '\t'+ '\t'.join(entity[:-1]) + '\n')
	i += 1
	if (i%1000) ==0:
		print (i)

print (num_features_c,num_features_t)
outfile1.close()