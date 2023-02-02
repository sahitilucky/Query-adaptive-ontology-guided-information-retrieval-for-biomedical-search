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