from Utils import *
#TASKS DONE
#1.Total corpus statistics - context words frequencies for both cases
#2.Word IDFs
#3.Type frequencies.
#generate list with positive and negative list of docs.

s = 'This, sginf.cdsds:'
print (preprocess(s))

trec_pmids = []
with open('../2004_TREC_PMID_list', 'r') as infile:
	for line in infile:
		trec_pmids += [line.strip()]
print (trec_pmids[:10])
trec_pmids_dict = dict(zip(trec_pmids, range(len(trec_pmids))))


print ('loading....')
#get documents content
corpus = {}
with open('../data_files/bioconcepts2pubtator_corpus.json', 'r') as infile:
    corpus = json.load(infile)
print ('loading done....')

doc_entities = {}
i = 0 
inputfile = '../data_files/bioconcepts2pubtator_trec_with_disgenet_ids_smaller'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num in trec_pmids_dict:
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
#print ('dumping')
#with open('../data_files/disgenet_doc_entities.json', 'w') as outfile:
#	json.dump(doc_entities, outfile)

print ('Number of docs of trec in bioconcepts data', len(doc_entities.keys()))

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
print (gene_to_disease.keys())
print (disease_to_gene.keys())

#[start, end, name, type, id1, id2,....]
doc_nums_hier = []
types_statistics = {}
context_statistics = {}
total_types_statistics = {}
total_context_statistics = {}
positive_document_dict = {}
negative_document_dict = {}
context_word_vocabulary = {}
iter = 0
for doc_num in doc_entities:
	if doc_num in corpus:
		all_entities = [entity[4] for entity in doc_entities[doc_num]]
		all_entities = [x for sub_list in all_entities for x in sub_list]
		all_entities = set(all_entities)
		for entity in doc_entities[doc_num]:
			required_entity_children = []
			if entity[3] == 'Gene':
				try:
					required_entity_children = gene_to_disease[entity[4][0]]
				except:
					print ('gene not there')
					pass
			else:
				for disease in entity[4]:
					try:
						required_entity_children += disease_to_gene[disease]
					except:
						print ('disease not there')
						pass
			left_context = corpus[doc_num][:int(entity[0])].split()[-5:]
			right_context = corpus[doc_num][int(entity[1]):].split()[:5]
			print (left_context)
			print (right_context)
			print (entity[2])
			print (corpus[doc_num])
			print (entity[3])
			print (required_entity_children)
			print (all_entities)
			context_words = preprocess(' '.join(left_context)).split()
			context_words += preprocess(' '.join(right_context)).split()
			context_words_dict = dict(zip(context_words,range(len(context_words))))
			if required_entity_children == []:
				continue
			if list(all_entities.intersection(set(required_entity_children))) != []:
				positive_document_dict[doc_num] = 1
				doc_nums_hier += [doc_num]
				try:
					types_statistics[entity[3]] += 1 
				except:
					types_statistics[entity[3]] = 1
				#print (entity[0])
				#print (entity[1])
				#print (corpus[doc_num][int(entity[0]): int(entity[1])])
				for word in context_words_dict:
					try:
						context_statistics[word] += 1
					except:
						context_statistics[word] = 1
			else:
				negative_document_dict[doc_num] = 1

			try:
				total_types_statistics[entity[3]] += 1 
			except:
				total_types_statistics[entity[3]] = 1
			
			for word in context_words_dict:
				try:
					total_context_statistics[word] += 1
				except:
					total_context_statistics[word] = 1
				context_word_vocabulary[word] = 1
	iter += 1
	if (iter%1000)==0:
		print (iter)
	if (iter==20):
		break
	if (iter%200000) == 0:
		#print ('Number of docs with parent and children until now:' , len(set(doc_nums_hier))) 
		print ('Saving Intermmediate result...')
		context_statistics_temp = sorted(context_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/disgenet_context_statistics_temp.txt', 'w') as outfile:
			for word in context_statistics_temp:
				outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_context_statistics[word[0]]) +'\n')

		types_statistics_temp = sorted(types_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/disgenet_types_statistics_temp.txt', 'w') as outfile:
			for word in types_statistics_temp:
				outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_types_statistics[word[0]])+ '\n')

		with open('../LC_related_data_2/disgenet_context_word_vocabulary_temp.json', 'w') as outfile:
			json.dump(context_word_vocabulary, outfile)


		with open('../LC_related_data_2/disgenet_positive_document_dict_temp.json', 'w') as outfile:
			json.dump(positive_document_dict, outfile)
			

		with open('../LC_related_data_2/disgenet_negative_document_dict_temp.json', 'w') as outfile:
			json.dump(negative_document_dict, outfile)
		
		with open('../LC_related_data_2/total_types_statistics_temp.json', 'w') as outfile:
			json.dump(total_types_statistics, outfile)
		with open('../LC_related_data_2/total_context_statistics_temp.json', 'w') as outfile:
			json.dump(total_context_statistics, outfile)

print ('Number of docs with parent and children:' , len(set(doc_nums_hier))) 
'''
context_statistics = sorted(context_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/disgenet_context_statistics.txt', 'w') as outfile:
	for word in context_statistics:
		outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_context_statistics[word[0]]) +'\n')

types_statistics = sorted(types_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/disgenet_types_statistics.txt', 'w') as outfile:
	for word in types_statistics:
		outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_types_statistics[word[0]])+ '\n')

with open('../LC_related_data_2/disgenet_context_word_vocabulary.json', 'w') as outfile:
	json.dump(context_word_vocabulary, outfile)


with open('../LC_related_data_2/disgenet_positive_document_dict.json', 'w') as outfile:
	json.dump(positive_document_dict, outfile)
	
with open('../LC_related_data_2/total_types_statistics.json', 'w') as outfile:
	json.dump(total_types_statistics, outfile)
with open('../LC_related_data_2/total_context_statistics.json', 'w') as outfile:
	json.dump(total_context_statistics, outfile)

with open('../LC_related_data_2/disgenet_negative_document_dict.json', 'w') as outfile:
	json.dump(negative_document_dict, outfile)
'''	
#whole corpus 
