from Utils import *
'''
corpus_pmid_list = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
	for line in infile:
		corpus_pmid_list += [line.strip()]
corpus_pmid_list_dict = dict(zip(corpus_pmid_list, range(len(corpus_pmid_list))))
print (len(corpus_pmid_list))
doc_entities = {}
i = 0 
inputfile = '../data_files/bioconcepts2pubtator_trec_disgenet_ids_abstract_mentions'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num in corpus_pmid_list_dict:
			#print (line)
			if line_list[4] == 'Disease':
				#entity_list = [line_list[-1][5:]]
				entity_list = ast.literal_eval(line_list[-1])   #[1:-1].split(',')
			elif line_list[4] == 'Gene':
				entity_list = [line_list[-1]]
			else:
				continue
			try:
				doc_entities[doc_num] += entity_list
			except:
				doc_entities[doc_num] = entity_list 
		else:
			print doc_num
		i += 1
		if (i%100000) == 0:
			print (i)
print (len(doc_entities.keys()))
with open('../data_files/doc_abstracts_disgenet_entities.txt' , 'w') as outfile:
	for doc_num in doc_entities:
		outfile.write(doc_num+'\t' + ';'.join(doc_entities[doc_num]) + '\n')
'''

corpus_pmid_list = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
	for line in infile:
		corpus_pmid_list += [line.strip()]
corpus_pmid_list_dict = dict(zip(corpus_pmid_list, range(len(corpus_pmid_list))))
print (len(corpus_pmid_list))
doc_entities = {}
i = 0 
doc_nums = []
inputfile = '../data_files/bioconcepts2pubtator_trec_sample_abstract_mentions'
with open(inputfile, 'r') as infile:
	for line in infile:
		doc_nums += [line.strip().split('\t')[0]]
		if (len(line.strip().split('\t')) < 6):
			#print (line)
			#print ('coming here')
			continue
		elif (line.strip().split('\t')[5][:5] == 'MESH:'):
			doc_num = line.strip().split('\t')[0]
			try:
				doc_entities[doc_num] += [line.strip().split('\t')[5][5:]]
			except:
				doc_entities[doc_num] = [line.strip().split('\t')[5][5:]]
		else:
			continue
		i += 1
		if (i%100000) == 0:
			print (i)
doc_nums = list(set(doc_nums))
print len(doc_nums)
print (len(doc_entities.keys()))
with open('../data_files/doc_abstracts_pubtator_entities.txt' , 'w') as outfile:
	for doc_num in doc_entities:
		outfile.write(doc_num+'\t' + ';'.join(doc_entities[doc_num]) + '\n')

