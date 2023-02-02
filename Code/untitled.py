abstract_doc_nums = {}
with open('../data_files/bioconcepts2pubtator_trec_sample_abstract_mentions', 'r') as infile:
	for line in infile:
		doc_num = line.strip().split('\t')[0]
		abstract_doc_nums[doc_num] = 1

with open('../data_files/bioconcepts2pubtator_trec_sample_title_mentions', 'r') as infile:
	with open('../data_files/bioconcepts2pubtator_trec_sample_new_title_mentions', 'w') as outfile:
		for line in infile:
			doc_num = line.strip().split('\t')[0]
			if doc_num in abstract_doc_nums:
				outfile.write(line)
			else:
				print doc_num
print (len(abstract_doc_nums.keys()))				
