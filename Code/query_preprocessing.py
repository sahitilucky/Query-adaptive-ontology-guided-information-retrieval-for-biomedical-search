from Utils import *

queries = []
with open('../data_files/2004_adhoc_topics.txt', 'r') as infile:
	query_id = ''
	query_need = ''
	query_context = ''
	for line in infile:
		if ('<ID>' in line) and ('</ID>' in line):
			query_id = int(line.strip().split('<ID>')[1].split('</ID>')[0])
			queries += [{}]
		if ('<NEED>' in line) and ('</NEED>' in line):
			queries[query_id-1]['need'] = line.strip().split('<NEED>')[1].split('</NEED>')[0]
		if ('<TITLE>' in line) and ('</TITLE>' in line):
			queries[query_id-1]['title'] = line.strip().split('<TITLE>')[1].split('</TITLE>')[0]
		if ('<CONTEXT>' in line) and ('</CONTEXT>' in line):
			#print 'coming here'
			queries[query_id-1]['context'] = line.strip().split('<CONTEXT>')[1].split('</CONTEXT>')[0]


for query in queries:
	print query
	query['title'] = preprocess(query['title'])
	query['need'] = preprocess(query['need'])
	query['context'] = preprocess(query['context'])

with open('../BM25-master/text/Trec_2004_full_queries.txt', 'w') as outfile:
	for query in queries:
		outfile.write(query['title'] + ' ' + query['need']  +' ' + query['context'] + '\n') 
		#outfile.write(query['need'] + '\n')


#with open('../data_files/queries.json', 'w') as outfile:
