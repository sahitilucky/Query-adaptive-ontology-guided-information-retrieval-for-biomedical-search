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
context_word_vocabulary_dict = dict(zip(context_word_vocabulary.keys(), range(1,len(context_word_vocabulary.keys())+1)))

print ('Total context words:' ,len(context_word_vocabulary.keys()))
#consider only top frequency context words
sorted_context_words = sorted(context_statistics.items(), key = lambda l : l[1][1], reverse =True)[:8000]
sorted_context_words = [x[0] for x in sorted_context_words]
context_word_vocabulary_dict = dict(zip(sorted_context_words, range(len(sorted_context_words))))
print ('New total context words:' ,len(context_word_vocabulary_dict.keys()))
print ('Save selected features indeces')
with open('../LC_related_data_2/disgenet_types_index_dict.json', 'w') as outfile:
	json.dump(types_index_dict,outfile)
with open('../LC_related_data_2/disgenet_context_word_vocabulary_dict.json', 'w') as outfile:
	json.dump(context_word_vocabulary_dict,outfile)
#save those dicts
'''
types_index_dict = json.load(open('../LC_related_data_2/disgenet_types_index_dict.json','r'))
context_word_vocabulary_dict = json.load(open('../LC_related_data_2/disgenet_context_word_vocabulary_dict.json','r'))
'''
positive_docs = positive_document_dict.keys()
positive_docs = random.sample(positive_docs, min(len(positive_docs),10000))
negative_docs = negative_document_dict.keys()
negative_docs = random.sample(negative_docs, min(len(negative_docs),10000))
sample_list = list(set(negative_docs+positive_docs+corpus_pmid_list))
sample_list_dict = dict(zip(sample_list, range(len(sample_list))))
print ('Final number of docs in the sample:' , len(sample_list))
with open('../LC_related_data_2/Disgenet_selected_docs.json', 'w') as outfile:
	json.dump(sample_list_dict, outfile)
'''
sample_list_dict = {}
with open('../LC_related_data_2/Mesh_selected_docs.json', 'r') as infile:
	sample_list_dict = json.load(infile)
sample_list = [0]*len(sample_list_dict.keys())
for s in sample_list_dict:
	sample_list[sample_list_dict[s]] = s

doc_entities = {}
i = 0 
inputfile = '../data_files/bioconcepts2pubtator_trec_with_disgenet_ids_smaller'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num in sample_list_dict:
			try:
				doc_entities[doc_num] += [line_list[1:]] 
			except:
				doc_entities[doc_num] = [line_list[1:]] 
			if line_list[4] == 'Disease':
				doc_entities[doc_num][-1][4] = ast.literal_eval(line_list[-1])   #[1:-1].split(',')
			else:
				if ';' in line_list[-1]:
					line_list[-1] = line_list[-1].split(';')[0]
				doc_entities[doc_num][-1][4] = [line_list[-1]]
		i += 1
		if (i%100000) == 0:
			print (i)

gene_to_disease = {}
disease_to_gene = {}
with open('../data_files/gene_disease_network.csv', 'r') as infile:
	for line in infile:
		gene = line.split(',')[1]
		disease = line.split(',')[2]
		score = line.strip().split(',')[3]
		if float(score)<=0.05:
			continue
		try:
			gene_to_disease[gene] += [disease]
		except:
			gene_to_disease[gene] = [disease]
		try:
			disease_to_gene[disease] += [gene]
		except:
			disease_to_gene[disease] = [gene]

#create training data
word_idf_list = context_word_idfs.values()
average_word_idf = float(sum(word_idf_list))/float(len(word_idf_list))
outfile1 = open('../LC_related_data_2/disgenet_positive_training_data_new_features.txt', 'w')
outfile2 = open('../LC_related_data_2/disgenet_negative_training_data_new_features.txt', 'w')
i = 0
num_features_c = len(context_word_vocabulary_dict.keys())
num_features_t = len(types_index_dict.keys())
print (num_features_c,num_features_t)
for doc_num in sample_list:
	if doc_num not in doc_entities:
		continue
	all_entities = [entity[4] for entity in doc_entities[doc_num]]
	all_entities = [x for sub_list in all_entities for x in sub_list]
	all_entities = set(all_entities)
	for entity in doc_entities[doc_num]:
		required_entity_children = []
		if entity[3] == 'Gene':
			try:
				required_entity_children = gene_to_disease[entity[4][0]]
			except:
				pass
		else:
			for disease in entity[4]:
				try:
					required_entity_children += disease_to_gene[disease]
				except:
					pass
			if required_entity_children == []:
				continue
		#left_context = corpus[doc_num][:int(entity[0])].split()[-5:]
		#right_context = corpus[doc_num][int(entity[1]):].split()[:5]
		#print left_context
		#print right_context
		#print entity[2]
		#print corpus[doc_num]
		#context_words = preprocess(' '.join(left_context)).split()
		#context_words += preprocess(' '.join(right_context)).split()
		#context_words_dict = dict(zip(context_words,range(len(context_words))))
		#print (all_entities)
		#print (required_entity_children) 
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
		if list(all_entities.intersection(set(required_entity_children))) != []:
			final_feature_string = ','.join([str(s) for s in final_feature])
			outfile1.write(final_feature_string + '\n')
		else:
			final_feature_string = ','.join([str(s) for s in final_feature])
			outfile2.write(final_feature_string + '\n')	
	i += 1
	if (i%1000) ==0:
		print (i)
		print (len(final_feature))
print (num_features_c,num_features_t)
outfile1.close()
outfile2.close()

