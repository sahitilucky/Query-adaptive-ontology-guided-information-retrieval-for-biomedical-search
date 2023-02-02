from Utils import *
'''
trec_pmids = []
with open('../2004_TREC_PMID_list', 'r') as infile:
    for line in infile:
        trec_pmids += [line.strip()]
print (trec_pmids[:10])
trec_pmids_dict = dict(zip(trec_pmids, range(len(trec_pmids))))

#pubtator input files
infile = open('../bioconcepts2pubtator_offsets', 'r')
line = infile.readline()
num_pmids = 0
outfile = open('../bioconcepts2pubtator_offsets_trec_pmids', 'w')
while (line != None) and (line.strip()!=''):
    lines = []
    lines += [line]
    pmid = lines[0].strip().split('|')[0]
    lines += [infile.readline()]
    line = infile.readline()
    i = 0
    while line.strip()!= '':
        lines += [line]
        line = infile.readline()
    num_pmids += 1
    if pmid in trec_pmids_dict:
        print (pmid)
        outfile.write(''.join(lines))
        outfile.write('\n')
    line = infile.readline()
    if (num_pmids%10000 ==0):
        print (num_pmids)
print(num_pmids)
infile.close()
'''
corpus_pmid_list = []
with open('../jinfeng_data/data/jinfeng/corpus_pmid.txt', 'r') as infile:
    for line in infile:
        corpus_pmid_list += [line.strip()]
trec_pmids = []
with open('../2004_TREC_PMID_list', 'r') as infile:
    for line in infile:
        trec_pmids += [line.strip()]

sample = corpus_pmid_list 
#trec_pmids_sample = random.sample(trec_pmids, 50000-len(corpus_pmid_list))
#sample = list(set(trec_pmids_sample+sample))

print ('Final sample length', len(sample)) 
#with open('../data_files/sample_docs_for_Of.txt', 'w') as outfile:
#    for doc in sample:
#        outfile.write(doc+'\n') 
sample_dict = dict(zip(sample, range(len(sample)))) 
#gene pubtator input files

corpus = {}
infile = open('../data_files/bioconcepts2pubtator_offsets_trec_pmids', 'r')
#outfile = open('../bioconcepts2pubtator_only_offsets_trec', 'w')
#outfile1 = open('../data_files/bioconcepts2pubtator_trec_sample_title_mentions', 'w')
#outfile2 = open('../data_files/bioconcepts2pubtator_trec_sample_abstract_mentions', 'w')
line = infile.readline()
i = 0
while (line != None) and (line.strip()!=''):
    pmid = line.strip().split('|')[0]
    title = line.strip().split('|')[2]
    line = infile.readline()
    abstract = line.strip().split('|')[2]
    if pmid in sample_dict:
        corpus[pmid] = {}
        corpus[pmid]['title'] = title
        corpus[pmid]['abstract'] = abstract
    line = infile.readline()
    while line.strip()!= '':
        if pmid in sample_dict:
            start_ind = int(line.strip().split('\t')[1])
            end_ind = int(line.strip().split('\t')[2])
            #if start_ind < len(title):
            #    outfile1.write(line)
            #else:
            #    outfile2.write(line)
    	#outfile.write(line)
    	line = infile.readline()
    line = infile.readline()
    i += 1
    if (i%1000)==0:
    	print (i)
print (i)

with open('../data_files/bioconcepts2pubtator_trec_sample_title_abstracts_corpus.json', 'w') as outfile:
    json.dump(corpus, outfile)

