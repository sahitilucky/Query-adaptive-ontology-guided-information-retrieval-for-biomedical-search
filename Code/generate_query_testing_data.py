from Utils import *
#TASKS
#1.Sample points accordingly - get document list with positives, get negative list as well, include get document list of those 12575 in sample
#Get word embeddings
#Generate training data for the sampled points

#loadinf input files
print ('loading....')
context_statistics = {}
with open('../LC_related_data_2/context_statistics.txt', 'r') as infile:
	for line in infile:
		[word,pos_freq,total_freq]= line.strip().split('\t')
		context_statistics[word] = [float(pos_freq),float(total_freq)]

types_statistics ={}
with open('../LC_related_data_2/types_statistics.txt', 'r') as infile:
	for line in infile:
		[word,pos_freq,total_freq]= line.strip().split('\t')
		types_statistics[word] = [float(pos_freq),float(total_freq)]
other_types_statistics = {}
with open('../LC_related_data_2/other_types_statistics.txt', 'r') as infile:
	for line in infile:
		[word,pos_freq,total_freq]= line.strip().split('\t')
		other_types_statistics[word] = [float(pos_freq),float(total_freq)]
specific_types_statistics = {}
mesh_id_to_name = {}
with open('../LC_related_data_2/specific_types_statistics.txt', 'r') as infile:
	for line in infile:
		[mesh_id,name,pos_freq,total_freq]= line.strip().split('\t')
		specific_types_statistics[mesh_id] = [float(pos_freq),float(total_freq)]
		mesh_id_to_name[mesh_id] = name

context_word_vocabulary = {}
with open('../LC_related_data_2/context_word_vocabulary.json', 'r') as infile:
	context_word_vocabulary = json.load(infile)

positive_document_dict = {}
with open('../LC_related_data_2/positive_document_dict.json', 'r') as infile:
	positive_document_dict = json.load(infile)
	
negative_document_dict = {}
with open('../LC_related_data_2/negative_document_dict.json', 'r') as infile:
	negative_document_dict = json.load(infile)
'''
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

types_index_dict = json.load(open('../LC_related_data_2/types_index_dict.json', 'r'))
other_types_index_dict = json.load(open('../LC_related_data_2/other_types_index_dict.json', 'r'))
specific_types_index_dict = json.load(open('../LC_related_data_2/specific_types_index_dict.json','r'))
context_word_vocabulary_dict =  json.load(open('../LC_related_data_2/context_word_vocabulary_dict.json', 'r'))

#entity 'C' to mesh 'D'
entity_to_mesh_links = {}
with open('../jinfeng_data/data/xuan/entity_mh_links.json', 'r') as infile:
	entity_to_mesh_links = json.load(infile) 

#bioconcept input files - getting local context ad bigger type
doc_entities = {}
i = 0 
#inputfile = '../jinfeng_data/data/jinfeng/pubtator_mention_2_meshid.txt'
inputfile = '../data_files/bioconcepts2pubtator_mesh_disgent_title_mentions.txt'
query_ids = []
query_ids_dict = {}
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num not in query_ids_dict:
			query_ids += [doc_num]
			query_ids_dict[doc_num] = 1
		if line_list[-1][:5] == 'MESH:':
			if doc_num in doc_entities:
				doc_entities[doc_num] += [line_list[1:]] 
			else:
				doc_entities[doc_num] = [line_list[1:]]
			entityid = line_list[-1] 
			entityid = entityid[5:]
			if (entityid[0] == 'C'):
				if entityid in entity_to_mesh_links:
					doc_entities[doc_num][-1][4] = entity_to_mesh_links[entityid] 
				else:
					#print ('gone')
					doc_entities[doc_num][-1][4] = [entityid]
			else:
				doc_entities[doc_num][-1][4] = [entityid] 
		i += 1
		if (i%100) == 0:
			print (i)
print (doc_entities.keys())

entity_other_types = {}
entity_specific_types = {}
with open('../jinfeng_data/data/xuan/midpath_mh.txt', 'r') as infile:
	for line in infile:
		mesh_node = line.split(':::')[0]
		entity_other_types[mesh_node] = line.strip().split('::')[1][1:]
		entity_specific_types[mesh_node] = line.strip().split('::')[2]

word_idf_list = context_word_idfs.values()
average_word_idf = float(sum(word_idf_list))/float(len(word_idf_list))
#create training data
outfile1 = open('../LC_related_data_2/trec_title_query_mesh_testing_data.txt', 'w')
outfile2 = open('../LC_related_data_2/trec_title_query_mesh_testing_data_order.txt', 'w')
i = 0
num_features_c = len(context_word_vocabulary_dict.keys())
num_features_s = len(specific_types_index_dict.keys())
num_features_t = len(types_index_dict.keys())
num_features_o = len(other_types_index_dict.keys())
print (num_features_c,num_features_t, num_features_o,num_features_s)
for doc_num in query_ids:
	doc_num = str(doc_num)
	if doc_num in doc_entities:
		for entity in doc_entities[doc_num]:
			other_types =[]
			specific_types = []
			for parent in entity[4]:
				try:
					other_types += [entity_other_types[parent]]
				except:
					pass
				try:
					specific_types += [entity_specific_types[parent]]
				except:
					pass
			other_types_dict = dict(zip(other_types,range(len(other_types))))
			specific_types_dict = dict(zip(specific_types,range(len(specific_types))))
			#print (all_entities)
			#print (required_entity_children) 
			feature_c = [0]*(num_features_c)
			feature_c_2 = [0]*(num_features_c)
			feature_c_3 = [0]*(num_features_c)
			feature_o = [0]*(num_features_o)
			feature_s = [0]*(num_features_s)
			feature_t = [0]*(num_features_t)
			try:
				feature_t[types_index_dict[entity[3]]] = float(types_statistics[entity[3]][0])/float(types_statistics[entity[3]][1])
			except KeyError:
				pass
			for other_type in other_types_dict:
				try:
					feature_o[other_types_index_dict[other_type]] = float(other_types_statistics[other_type][0])/float(other_types_statistics[other_type][1])
				except:
					pass
			for specific_type in specific_types_dict:
				try:
					feature_s[specific_types_index_dict[specific_type]] = float(specific_types_statistics[specific_type][0])/float(specific_types_statistics[specific_type][1])
				except:
					pass
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
			feature_c.extend(feature_o)
			feature_c.extend(feature_s)
			final_feature = feature_c
			final_feature_string = ','.join([str(s) for s in final_feature])
			outfile1.write(final_feature_string + '\n')
			outfile2.write(doc_num + '\t'+ '\t'.join(entity[:-1]) + '\n')
	i += 1
	if (i%1000) ==0:
		print (i)

print (num_features_c,num_features_t, num_features_o,num_features_s)
outfile1.close()