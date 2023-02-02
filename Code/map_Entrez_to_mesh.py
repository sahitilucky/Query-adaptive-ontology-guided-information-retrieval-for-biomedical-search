from Utils import *
'''
entrez_id_to_mesh_id = {}
i = 0
with open('../data_files/entrez_mesh_mapping_human.csv') as infile:
	for line in infile:
		entrez_id = line.split(',')[1]
		mesh_id = line.split(',')[3].split('"')[1]
		i+=1
		if i<10:
			print (entrez_id,mesh_id)
		if mesh_id[0] == 'C':
			print ('it is there')
		if entrez_id in entrez_id_to_mesh_id:
			entrez_id_to_mesh_id[entrez_id] += [mesh_id]
		else:
			entrez_id_to_mesh_id[entrez_id] = [mesh_id]
		if (i%1000000) ==0 :
			print (i)
i = 0	
with open('../data_files/entrez_mesh_mapping_mice.csv') as infile:
	for line in infile:
		entrez_id = line.split(',')[1]
		mesh_id = line.split(',')[3].split('"')[1]
		i+=1
		if i<10:
			print (entrez_id,mesh_id)
		if mesh_id[0] == 'C':
			print ('it is there')
		if entrez_id in entrez_id_to_mesh_id:
			entrez_id_to_mesh_id[entrez_id] += [mesh_id]
		else:
			entrez_id_to_mesh_id[entrez_id] = [mesh_id]
		if (i%1000000) == 0:
			print (i)

print ('Total number of entrez ids', len(entrez_id_to_mesh_id.keys()))
lengths = []
for entrez_id in entrez_id_to_mesh_id:
	entrez_id_to_mesh_id[entrez_id] = list(set(entrez_id_to_mesh_id[entrez_id]))
	lengths += [len(entrez_id_to_mesh_id[entrez_id])]

avg_length = float(sum(lengths))/float(len(lengths))
print ('Average length:' , avg_length)
print (lengths[:100])
i = 0
inputfile = '../data_files/bioconcepts2pubtator_only_offsets_trec'
outputfile = '../data_files/bioconcepts2pubtator_trec_with_mesh_ids_smaller'
with open(inputfile, 'r') as infile:
	with open(outputfile, 'w') as outfile:
		for line in infile:
			if (len(line.strip().split('\t')) < 6):
				continue
			elif (line.strip().split('\t')[4] == 'Gene'):
				#print (line)
				geneid = line.strip().split('\t')[5]
				if ('(') in geneid:
					geneid = geneid.split('(')[0]
				rem_line = '\t'.join(line.strip().split('\t')[:5])
				if geneid in entrez_id_to_mesh_id:
					expansion_mesh_ids = random.sample(entrez_id_to_mesh_id[geneid], min(10, len(entrez_id_to_mesh_id[geneid])))
					outfile.write(rem_line + '\t' + 'MESH:' + str(expansion_mesh_ids) + '\n')
			elif (line.strip().split('\t')[5][:5] == 'MESH:'):
				outfile.write(line)
			else:
				continue
			i += 1
			if (i%100000) == 0:
				print (i)
'''

#mapping mesh id to UMLS for disgenet.
#entity 'C' to mesh 'D'
entity_to_mesh_links = {}
with open('../jinfeng_data/data/xuan/entity_mh_links.json', 'r') as infile:
	entity_to_mesh_links = json.load(infile) 

mesh_id_to_umls_id = {}
i = 0
with open('../data_files/umls_mesh_mapping.csv', 'r') as infile:
	for line in infile:
		mesh_id = line.strip().split(',')[0]
		umls_id = line.strip().split(',')[1]
		i+=1
		if i<10:
			print (mesh_id,umls_id)
		try:
			mesh_id_to_umls_id[mesh_id] += [umls_id]
		except:
			mesh_id_to_umls_id[mesh_id] = [umls_id]
		if mesh_id[0] == 'C':
			if mesh_id in entity_to_mesh_links:
				all_mesh_ids = entity_to_mesh_links[mesh_id]
				for mesh_id in all_mesh_ids:
					try:
						mesh_id_to_umls_id[mesh_id] += [umls_id]
					except:
						mesh_id_to_umls_id[mesh_id] = [umls_id]
			

lengths = []
for mesh_id in mesh_id_to_umls_id:
	mesh_id_to_umls_id[mesh_id] = list(set(mesh_id_to_umls_id[mesh_id]))
	lengths += [len(mesh_id_to_umls_id[mesh_id])]

avg_length = float(sum(lengths))/float(len(lengths))
print ('Average length:' , avg_length)
print (lengths[:100])

inputfile = '../data_files/bioconcepts2pubtator_trec_sample_abstract_mentions'
outputfile = '../data_files/bioconcepts2pubtator_trec_disgenet_ids_abstract_mentions'
with open(inputfile, 'r') as infile:
	with open(outputfile, 'w') as outfile:
		for line in infile:
			if (len(line.strip().split('\t')) < 6):
				continue
			elif (line.strip().split('\t')[4] == 'Gene'):
				geneid = line.strip().split('\t')[5]
				if ('(') in geneid:
					geneid = geneid.split('(')[0]
				rem_line = '\t'.join(line.strip().split('\t')[:5])
				outfile.write(rem_line + '\t' + geneid + '\n')
			elif (line.strip().split('\t')[5][:5] == 'MESH:'):
				if (line.strip().split('\t')[4] == 'Disease'):
					mesh_id = line.strip().split('\t')[5][5:]
					rem_line = '\t'.join(line.strip().split('\t')[:5])
					if mesh_id in mesh_id_to_umls_id:
						outfile.write(rem_line + '\t' + str(mesh_id_to_umls_id[mesh_id]) + '\n')
					else:
						if mesh_id in entity_to_mesh_links:
							all_mesh_ids = entity_to_mesh_links[mesh_id]
							umls_ids = [mesh_id_to_umls_id[x] for x in all_mesh_ids if x in mesh_id_to_umls_id]
							umls_ids = list(set([x for sub_list in umls_ids for x in sub_list]))
							for umls_id in umls_ids:
								outfile.write(rem_line + '\t' + str(umls_ids) + '\n')
			else:
				continue
			i += 1
			if (i%100000) == 0:
				print (i)
