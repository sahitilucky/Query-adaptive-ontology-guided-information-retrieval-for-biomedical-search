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

#get documents content
corpus = {}
with open('../data_files/bioconcepts2pubtator_corpus.json', 'r') as infile:
    corpus = json.load(infile)

corpus_pmid_list = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
	for pmid in infile:
		corpus_pmid_list += [pmid.strip()]

context_word_idfs = {}
with open('../data_files/bioconcepts2pubtator_corpus_idfs.json', 'r') as infile:
    context_word_idfs = json.load(infile)
print ('loading done....')
'''
types_index_dict = dict(zip(types_statistics.keys(), range(len(types_statistics.keys()))))
other_types_index_dict = dict(zip(other_types_statistics.keys(), range(len(other_types_statistics.keys()))))
specific_types_index_dict = dict(zip(specific_types_statistics.keys(), range(len(specific_types_statistics.keys()))))
context_word_vocabulary_dict = dict(zip(context_word_vocabulary.keys(), range(len(context_word_vocabulary.keys()))))

print ('Total context words:' ,len(context_word_vocabulary.keys()))
#consider only top frequency context words
sorted_context_words = sorted(context_statistics.items(), key = lambda l : l[1][1], reverse =True)[:8000]
sorted_context_words = [x[0] for x in sorted_context_words]
context_word_vocabulary_dict = dict(zip(sorted_context_words, range(len(sorted_context_words))))
print ('New Total context words:' ,len(context_word_vocabulary_dict.keys()))
print ('Save selected features indeces')
with open('../LC_related_data_2/types_index_dict.json', 'w') as outfile:
	json.dump(types_index_dict,outfile)
with open('../LC_related_data_2/other_types_index_dict.json', 'w') as outfile:
	json.dump(other_types_index_dict,outfile)
with open('../LC_related_data_2/specific_types_index_dict.json', 'w') as outfile:
	json.dump(specific_types_index_dict,outfile)
with open('../LC_related_data_2/context_word_vocabulary_dict.json', 'w') as outfile:
	json.dump(context_word_vocabulary_dict,outfile)
#save those dicts
'''
types_index_dict = json.load(open('../LC_related_data_2/types_index_dict.json', 'r'))
other_types_index_dict = json.load(open('../LC_related_data_2/other_types_index_dict.json', 'r'))
specific_types_index_dict = json.load(open('../LC_related_data_2/specific_types_index_dict.json','r'))
context_word_vocabulary_dict =  json.load(open('../LC_related_data_2/context_word_vocabulary_dict.json', 'r'))
'''
positive_docs = positive_document_dict.keys()
positive_docs = random.sample(positive_docs, 10000)
negative_docs = negative_document_dict.keys()
negative_docs = random.sample(negative_docs, 10000)
sample_list = list(set(negative_docs+positive_docs+corpus_pmid_list))
sample_list_dict = dict(zip(sample_list, range(len(sample_list))))
print ('Final number of docs in the sample:' , len(sample_list))
with open('../LC_related_data_2/Mesh_selected_docs.json', 'w') as outfile:
	json.dump(sample_list_dict, outfile)
'''
sample_list_dict = {}
with open('../LC_related_data_2/Mesh_selected_docs.json', 'r') as infile:
	sample_list_dict = json.load(infile)
sample_list = [0]*len(sample_list_dict.keys())
for s in sample_list_dict:
	sample_list[sample_list_dict[s]] = s

#entity 'C' to mesh 'D'
entity_to_mesh_links = {}
with open('../jinfeng_data/data/xuan/entity_mh_links.json', 'r') as infile:
	entity_to_mesh_links = json.load(infile) 

#bioconcept input files - getting local context ad bigger type
doc_entities = {}
i = 0 
#inputfile = '../jinfeng_data/data/jinfeng/pubtator_mention_2_meshid.txt'
inputfile = '../data_files/bioconcepts2pubtator_trec_with_mesh_ids_smaller'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num in sample_list_dict:
			try:
				doc_entities[doc_num] += [line_list[1:]] 
			except:
				doc_entities[doc_num] = [line_list[1:]]

			if line_list[4] == 'Gene':
				entity_list = line_list[-1][5:]
				doc_entities[doc_num][-1][4] = ast.literal_eval(entity_list)   #[1:-1].split(',') 
			else:
				entityid = line_list[-1] 
				entityid = entityid[5:]
				if (entityid[0] == 'C'):
					try:
						doc_entities[doc_num][-1][4] = entity_to_mesh_links[entityid] 
					except:
						#print ('gone')
						doc_entities[doc_num][-1][4] = [entityid]
				else:
					doc_entities[doc_num][-1][4] = [entityid] 
		i += 1
		if (i%100000) == 0:
			print (i)

print ('Doc entities:', len(doc_entities.keys()))
#get other types and specific types
entity_other_types = {}
entity_specific_types = {}
with open('../jinfeng_data/data/xuan/midpath_mh.txt', 'r') as infile:
	for line in infile:
		mesh_node = line.split(':::')[0]
		entity_other_types[mesh_node] = line.strip().split('::')[1][1:]
		entity_specific_types[mesh_node] = line.strip().split('::')[2]

#get entity children
#get parent child relation
entity_children = {}
with open('../jinfeng_data/data/xuan/midpath_mh_parent_child.txt', 'r') as infile:
	for line in infile:
		parent = line.strip().split()[0]
		child = line.strip().split()[1]
		if parent in entity_children:
			entity_children[parent] += [child]
		else:
			entity_children[parent] = [child]

#get_second_level_children
new_entity_children = {}
for parent in entity_children:
	first_children = entity_children[parent][:]
	new_entity_children[parent] = first_children
	for child in first_children:
		try:
			new_entity_children[parent] += entity_children[child]
		except:
			pass
	new_entity_children[parent] = list(set(new_entity_children[parent]))
entity_children = new_entity_children
#get third level children
new_entity_children = {}
for parent in entity_children:
	first_second_children = entity_children[parent][:]
	new_entity_children[parent] = first_second_children
	for child in first_second_children:
		try:
			new_entity_children[parent] += entity_children[child]
		except:
			pass
	new_entity_children[parent] = list(set(new_entity_children[parent]))
entity_children = new_entity_children

word_idf_list = context_word_idfs.values()
average_word_idf = float(sum(word_idf_list))/float(len(word_idf_list))
#create training data
outfile1 = open('../LC_related_data_2/positive_training_data_new_features.txt', 'w')
outfile2 = open('../LC_related_data_2/negative_training_data_new_features.txt', 'w')
i = 0
num_features_c = len(context_word_vocabulary_dict.keys())
num_features_s = len(specific_types_index_dict.keys())
num_features_t = len(types_index_dict.keys())
num_features_o = len(other_types_index_dict.keys())
print (num_features_c,num_features_t, num_features_o,num_features_s)
positive_training_data = []
negative_training_data = []
for doc_num in sample_list:
	if doc_num not in doc_entities:
		continue
	all_entities = [entity[4] for entity in doc_entities[doc_num]]
	all_entities = set([x for sub_list in all_entities for x in sub_list])
	for entity in doc_entities[doc_num]:
		required_entity_children = []
		other_types =[]
		specific_types = []
		for parent in entity[4]:
			try:
				required_entity_children += entity_children[parent]
			except:
				pass
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
		if required_entity_children == []:
			continue
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
		if list(all_entities.intersection(set(required_entity_children))) != []:
			final_feature_string = ','.join([str(s) for s in final_feature])
			#positive_training_data += [final_feature_string]
			outfile1.write(final_feature_string + '\n')
		else:
			final_feature_string = ','.join([str(s) for s in final_feature])
			#negative_training_data += [final_feature_string]
			outfile2.write(final_feature_string + '\n')	
	i += 1
	if (i%1000) ==0:
		print (i)
		print (len(final_feature))
print (num_features_c,num_features_t, num_features_o,num_features_s)
print ('writing')
'''
for line in positive_training_data:
	outfile1.write(line + '\n')

for line in negative_training_data:
	outfile2.write(line + '\n')
'''
outfile1.close()
outfile2.close()

