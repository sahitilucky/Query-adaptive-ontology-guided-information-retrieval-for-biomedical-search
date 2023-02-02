from Utils import *
'''
i = 0
with open('../jinfeng_data/data/jinfeng/corpus.json', 'r') as infile:
	with open('../BM25-master/text/pmid_corpus.txt', 'w') as outfile:
		for line in infile:
			document = ast.literal_eval(line)
			pmid = document['pmid'] 
			content = preprocess(document['title'] + ' ' + document['abstract'])
			outfile.write('# ' + pmid + '\n')
			outfile.write(content + '\n')
			i += 1
			if (i%200) == 0:
				print i
'''
'''
corpus_pmid = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
	for line in infile:
		corpus_pmid += [line.strip()]
#Getting all documents
corpus = {}
add_to_abstract = False
with open('../2004_TREC_ASCII_MEDLINE_1', 'r') as infile:
	for line in infile:
		#print (line)
		#print (add_to_abstract)
		if 'PMID-' in line:
			add_to_abstract = False
			pmid = line.strip().split('PMID- ')[1]
			corpus[pmid] = {}
		elif 'AB  -' in line:
			#print 'coming here'
			add_to_abstract = True
			corpus[pmid]['abstract'] = line.strip().split('AB  - ')[1]
		elif 'AD  -' in line:
			#print (line)
			add_to_abstract = False
		elif 'TI  -' in line:
			add_to_abstract = False
			corpus[pmid]['title'] = line.strip().split('TI  - ')[1]
		else:
			if add_to_abstract == True:
				#print (line)
				#print (line.strip().split('    '))
				corpus[pmid]['abstract'] += ' ' + line.strip().split('      ')[0]

print len(corpus.keys())
for i in range(10):
	print (corpus[corpus.keys()[i]])
i = 0

with open('../data_files/trec_12575_title_abstracts.txt', 'w') as outfile:
	for pmid in corpus_pmid:
		if pmid in corpus:
			outfile.write(pmid + '\n')
			outfile.write(corpus[pmid]['title'] + '\n')
			if ('abstract') in corpus[pmid]:
				outfile.write(corpus[pmid]['abstract'] + '\n')
			else:
				outfile.write('\n')
'''
'''
corpus_id_docs = {}
with open('../jinfeng_data/data/jinfeng/doc_pubtator_entities.txt', 'r') as infile:
    for line in infile:
        if len(line.strip().split('\t'))==2:
            [doc_id,mesh_ids] = line.strip().split('\t')
            corpus_id_docs[doc_id] = mesh_ids.split(';')
        else:
            [doc_id] = line.strip().split('\t')
            corpus_id_docs[doc_id] = []
with open('../jinfeng_data/data/jinfeng/doc_pubmed_mh.txt', 'r') as infile:
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

mesh_id_dict_file = '../jinfeng_data/data/jinfeng/mh2id.txt'
mesh_id_dict = {}
with open(mesh_id_dict_file) as f:
    for line in f:
        l = line.strip().split('\t')
        mesh_id_dict[l[1]] = l[0]

with open('../data_files/doc_mesh_id_names.txt','w') as outfile:
	for doc_id in corpus_id_docs:
		outfile.write('# ' + doc_id + '\n')
		for mesh_id in corpus_id_docs[doc_id]:
			try:
				outfile.write(mesh_id + '\t' + mesh_id_dict[mesh_id] + '\n')
			except:
				outfile.write(mesh_id + '\n')
'''

'''
corpus_pmid = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
	for line in infile:
		corpus_pmid += [line.strip()]
#Getting all documents
corpus = {}
add_to_abstract = False
with open('../2004_TREC_ASCII_MEDLINE_1', 'r') as infile:
	for line in infile:
		#print (line)
		#print (add_to_abstract)
		if 'PMID-' in line:
			add_to_abstract = False
			pmid = line.strip().split('PMID- ')[1]
			corpus[pmid] = []
		elif 'MH  - ' in line:
			#print 'coming here'
			add_to_abstract = True
			corpus[pmid] += [line.strip().split('MH  - ')[1]]
		else:
			continue

mesh_id_dict_file = '../jinfeng_data/data/jinfeng/mh2id.txt'
mesh_id_dict = {}
mesh_name_id = {}
with open(mesh_id_dict_file) as f:
    for line in f:
        l = line.strip().split('\t')
        mesh_id_dict[l[1]] = l[0]
        mesh_name_id[l[0]] = l[1] 
#print (mesh_name_id.keys()[:100])
corpus_ids = {}
for pmid in corpus_pmid:
	if pmid in corpus:
		for mh in corpus[pmid]:
			mh = mh.lower()
			if ('/' in mh):
				mh = mh.strip().split('/')[0]
			mh = mh.replace('*', '')
			mh = mh.replace('(', '')
			mh = mh.replace(')', '')
			if mh in mesh_name_id:
				try:
					corpus_ids[pmid] += [mesh_name_id[mh]]
				except:
					corpus_ids[pmid] = [mesh_name_id[mh]]
			else:
				if ', ' in mh:
					mh2 = ', '.join(mh.split(', ')[1:]) + ' ' + mh.split(', ')[0]
					if mh2 in mesh_name_id:
						print (mh2)
						try:
							corpus_ids[pmid] += [mesh_name_id[mh2]]
						except:
							corpus_ids[pmid] = [mesh_name_id[mh2]]
				else:
					mh = regex.sub('', mh)
					if mh in mesh_name_id:
						print (mh)
						try:
							corpus_ids[pmid] += [mesh_name_id[mh]]
						except:
							corpus_ids[pmid] = [mesh_name_id[mh]]
print (len(corpus_ids.keys()))
with open('../data_files/doc_abstracts_pubmed_mh.txt', 'w') as outfile:
	for pmid in corpus_ids:
		outfile.write(pmid + '\t' + ';'.join(corpus_ids[pmid]) + '\n')
'''
doc_nums = []
with open('../data_files/bioconcepts2pubtator_trec_sample_new_title_mentions', 'r') as infile:
	for line in infile:
		doc_nums += [line.strip().split()[0]]
doc_nums = list(set(doc_nums))
print (len(doc_nums))
doc_num_dict = dict(zip(doc_nums, range(len(doc_nums)))) 

corpus = {}
with open('../data_files/bioconcepts2pubtator_trec_sample_title_abstracts_corpus.json', 'r') as infile:
	corpus = json.load(infile)
with open('../BM25-master/text/trec_sample_abstract_corpus.txt', 'w') as outfile1:
	with open('../BM25-master/text/trec_sample_queries.txt', 'w') as outfile2:
		for pmid in corpus:
			if pmid in doc_num_dict:
				title = preprocess(corpus[pmid]['title']) 
				outfile2.write('# ' + pmid + '\n')
				outfile2.write(title + '\n')
				abstract = preprocess(corpus[pmid]['abstract'])
				outfile1.write('# ' + pmid + '\n')
				outfile1.write(abstract + '\n')





