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

#entity 'C' to mesh 'D'
entity_to_mesh_links = {}
with open('../jinfeng_data/data/xuan/entity_mh_links.json') as infile:
	entity_to_mesh_links = json.load(infile) 

print ('loading....')
#get documents content
corpus = {}
with open('../data_files/bioconcepts2pubtator_corpus.json', 'r') as infile:
    corpus = json.load(infile)
print ('loading done....')

#bioconcept input files - getting local context ad bigger type
doc_entities = {}
i = 0 
#inputfile = '../jinfeng_data/data/jinfeng/pubtator_mention_2_meshid.txt'
inputfile = '../data_files/bioconcepts2pubtator_trec_with_mesh_ids_smaller'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num in trec_pmids_dict:
			try:
				doc_entities[doc_num] += [line_list[1:]] 
			except:
				doc_entities[doc_num] = [line_list[1:]]

			if line_list[4] == 'Gene':
				entity_list = line_list[-1][5:]
				doc_entities[doc_num][-1][4] = ast.literal_eval(entity_list)  #[1:-1].split(',')
			else:
				entityid = line_list[-1] 
				entityid = entityid[5:]
				if (entityid[0] == 'C'):
					try:
						doc_entities[doc_num][-1][4] = entity_to_mesh_links[entityid] 
					except:
						doc_entities[doc_num][-1][4] = [entityid]
				else:
					doc_entities[doc_num][-1][4] = [entityid] 
		i += 1
		if (i%100000) == 0:
			print (i)


#bioconcept input files - getting local context ad bigger type
'''
doc_entities = {}
i = 0 
#inputfile = '../jinfeng_data/data/jinfeng/pubtator_mention_2_meshid.txt'
inputfile = '../data_files/bioconcepts2pubtator_trec_with_mesh_ids'
with open(inputfile, 'r') as infile:
	for line in infile:
		line_list = line.strip().split('\t')
		doc_num = line_list[0]
		if doc_num not in trec_pmids_dict:
			continue
		#if len(line.strip().split('\t')) < 6:
			#print (line)
		#	continue
		entityid = line_list[-1]
		#if ('MESH:' != entityid[:5]):
		#	continue
		if doc_num in doc_entities:
			doc_entities[doc_num] += [line_list[1:]] 
		else:
			doc_entities[doc_num] = [line_list[1:]] 
		entityid = entityid[5:]
		#print (entityid)
		if (entityid[0] == 'C'):
			if entityid in entity_to_mesh_links:
				doc_entities[doc_num][-1][4] = entity_to_mesh_links[entityid] 
			else:
				#print ('gone')
				doc_entities[doc_num][-1][4] = [entityid]
		else:
			doc_entities[doc_num][-1][4] = [entityid] 
		i += 1
		if (i%100000) == 0:
			print (i)
		#if (i==1000000):
		#	break
print ('dumping')
with open('../data_files/doc_entities.json', 'w') as outfile:
	json.dump(doc_entities, outfile)
'''
#print ('loading')
#with open('../data_files/doc_entities.json', 'r') as infile:
#	doc_entities = json.load(infile)

print ('Number of docs of trec in bioconcepts data', len(doc_entities.keys()))

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

#get other types and specific types
entity_other_types = {}
entity_specific_types = {}
with open('../jinfeng_data/data/xuan/midpath_mh.txt', 'r') as infile:
	for line in infile:
		mesh_node = line.split(':::')[0]
		entity_other_types[mesh_node] = line.strip().split('::')[1][1:]
		entity_specific_types[mesh_node] = line.strip().split('::')[2]

#get mesh id names
mesh_name_to_id = {}
mesh_id_to_name = {}
with open('../jinfeng_data/data/xuan/mh2id.json', 'r') as infile:
	mesh_name_to_id = json.load(infile)
for name in mesh_name_to_id:
	mesh_id_to_name[mesh_name_to_id[name]] = name



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


#[start, end, name, type, id1, id2,....]
doc_nums_hier = []
types_statistics = {}
context_statistics = {}
other_types_statistics = {}
specific_types_statistics = {}
total_types_statistics = {}
total_context_statistics = {}
total_other_types_statistics = {}
total_specific_types_statistics = {}
positive_document_dict = {}
negative_document_dict = {}
context_word_vocabulary = {}
iter = 0
for doc_num in doc_entities:
	if doc_num in corpus:
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
			other_types = list(set(other_types))
			specific_types = list(set(specific_types))
			other_types_dict = dict(zip(other_types,range(len(other_types))))
			specific_types_dict = dict(zip(specific_types,range(len(specific_types))))
			#print (all_entities)
			#print (required_entity_children)
			left_context = corpus[doc_num][:int(entity[0])].split()[-5:]
			right_context = corpus[doc_num][int(entity[1]):].split()[:5]
			context_words = preprocess(' '.join(left_context)).split()
			context_words += preprocess(' '.join(right_context)).split()
			context_words_dict = dict(zip(context_words,range(len(context_words))))
			'''
			print left_context
			print right_context
			print entity[2]
			print corpus[doc_num]
			print other_types
			print specific_types
			print required_entity_children
			'''	
			if (required_entity_children == []):
				continue
			if list(all_entities.intersection(set(required_entity_children))) != []:
				#print ('there is intersection')
				#print (doc_num)
				positive_document_dict[doc_num] = 1
				doc_nums_hier += [doc_num]
				try:
					types_statistics[entity[3]] += 1 
				except:
					types_statistics[entity[3]] = 1
				for other_type in other_types_dict:
					try:
						other_types_statistics[other_type] += 1
					except:
						other_types_statistics[other_type] = 1
				for specific_type in specific_types_dict:
					try:
						specific_types_statistics[specific_type] += 1
					except:
						specific_types_statistics[specific_type] = 1
				#print (entity[0])
				#print (entity[1])
				#print (corpus[doc_num][int(entity[0]): int(entity[1])])
				for word in context_words_dict:
					if word in context_statistics:
						context_statistics[word] += 1
					else:
						context_statistics[word] = 1
			else:
				negative_document_dict[doc_num] = 1

			try:
				total_types_statistics[entity[3]] += 1 
			except:
				total_types_statistics[entity[3]] = 1
			for other_type in other_types_dict:
				try:
					total_other_types_statistics[other_type] += 1
				except:
					total_other_types_statistics[other_type] = 1
			for specific_type in specific_types_dict:
				try:
					total_specific_types_statistics[specific_type] += 1
				except:
					total_specific_types_statistics[specific_type] = 1

			for word in context_words_dict:
				try:
					total_context_statistics[word] += 1
				except:
					total_context_statistics[word] = 1
				context_word_vocabulary[word] = 1
	
	iter += 1
	if (iter%1000)==0:
		print (iter)
	if (iter%100000) == 0:
		#print ('Number of docs with parent and children until now:' , len(set(doc_nums_hier))) 
		print ('Saving Intermmediate result...')
		print ('Number of docs with parent and children:' , len(set(doc_nums_hier)))
		
		context_statistics_temp = sorted(context_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/context_statistics_temp.txt', 'w') as outfile:
			for word in context_statistics_temp:
				outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_context_statistics[word[0]]) +'\n')

		types_statistics_temp = sorted(types_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/types_statistics_temp.txt', 'w') as outfile:
			for word in types_statistics_temp:
				outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_types_statistics[word[0]])+ '\n')

		other_types_statistics_temp = sorted(other_types_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/other_types_statistics_temp.txt', 'w') as outfile:
			for word in other_types_statistics_temp:
				outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_other_types_statistics[word[0]])+ '\n')

		specific_types_statistics_temp = sorted(specific_types_statistics.items(), key = lambda l: l[1], reverse=True)
		with open('../LC_related_data_2/specific_types_statistics_temp.txt', 'w') as outfile:
			for word in specific_types_statistics_temp:
				outfile.write(word[0] + '\t'+ mesh_id_to_name[word[0]] + '\t' + str(word[1]) + '\t' +  str(total_specific_types_statistics[word[0]])+ '\n')

		with open('../LC_related_data_2/context_word_vocabulary_temp.json', 'w') as outfile:
			json.dump(context_word_vocabulary, outfile)


		with open('../LC_related_data_2/positive_document_dict_temp.json', 'w') as outfile:
			json.dump(positive_document_dict, outfile)
			

		with open('../LC_related_data_2/negative_document_dict_temp.json', 'w') as outfile:
			json.dump(negative_document_dict, outfile)

		with open('../LC_related_data_2/total_specific_types_statistics_temp.json', 'w') as outfile:
			json.dump(total_specific_types_statistics, outfile)
		
		with open('../LC_related_data_2/total_other_types_statistics_temp.json', 'w') as outfile:
			json.dump(total_other_types_statistics, outfile)

		with open('../LC_related_data_2/total_context_statistics_temp.json', 'w') as outfile:
			json.dump(total_context_statistics, outfile)

		with open('../LC_related_data_2/total_types_statistics_temp.json', 'w') as outfile:
			json.dump(total_types_statistics, outfile)

print ('Number of docs with parent and children:' , len(set(doc_nums_hier))) 

context_statistics = sorted(context_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/context_statistics.txt', 'w') as outfile:
	for word in context_statistics:
		outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_context_statistics[word[0]]) +'\n')

types_statistics = sorted(types_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/types_statistics.txt', 'w') as outfile:
	for word in types_statistics:
		outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_types_statistics[word[0]])+ '\n')

other_types_statistics = sorted(other_types_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/other_types_statistics.txt', 'w') as outfile:
	for word in other_types_statistics:
		outfile.write(word[0] + '\t' + str(word[1]) + '\t' +  str(total_other_types_statistics[word[0]])+ '\n')

specific_types_statistics = sorted(specific_types_statistics.items(), key = lambda l: l[1], reverse=True)
with open('../LC_related_data_2/specific_types_statistics.txt', 'w') as outfile:
	for word in specific_types_statistics:
		outfile.write(word[0] + '\t'+ mesh_id_to_name[word[0]] + '\t' + str(word[1]) + '\t' +  str(total_specific_types_statistics[word[0]])+ '\n')

with open('../LC_related_data_2/context_word_vocabulary.json', 'w') as outfile:
	json.dump(context_word_vocabulary, outfile)

with open('../LC_related_data_2/positive_document_dict.json', 'w') as outfile:
	json.dump(positive_document_dict, outfile)
	

with open('../LC_related_data_2/negative_document_dict.json', 'w') as outfile:
	json.dump(negative_document_dict, outfile)

with open('../LC_related_data_2/total_specific_types_statistics.json', 'w') as outfile:
	json.dump(total_specific_types_statistics, outfile)
		
with open('../LC_related_data_2/total_other_types_statistics.json', 'w') as outfile:
	json.dump(total_other_types_statistics, outfile)

with open('../LC_related_data_2/total_context_statistics.json', 'w') as outfile:
	json.dump(total_context_statistics, outfile)

with open('../LC_related_data_2/total_types_statistics.json', 'w') as outfile:
	json.dump(total_types_statistics, outfile)
#whole corpus 
