import re
import os
import seekpath
import numpy as np

def kpath(infile,grid_size,with_time_reversal,recipe,threshhold,symprec,angle_tolerence, supercell_matrix=np.eye(3)):
	file = open(infile,'r')
	POSCAR = file.readlines()

	#cell
	cell_matrix = POSCAR[2:5]
	cell = np.zeros(shape=(3,3))

	for i in range(len(cell_matrix)):
		cell_matrix0= np.array(cell_matrix[i].split())
		cell[i,:]  = (cell_matrix0.astype(np.float))*np.array(POSCAR[1].split()).astype(np.float)
	


	#positions
    # POSCAR index changed by Nicholas Pike from 5 -> 6 and from 7 -> 8 
    # Previously, POSCAR[5] referenced the atom names i.e. Na Cl and not the 
    # atom numbers
	atoms = np.array(POSCAR[6].split()).astype(np.int)
	positions_matrix = POSCAR[8:8+sum(atoms)]	
	positions = np.zeros(shape=(np.sum(atoms),3))

	for j in range(len(positions_matrix)):
		positions_matrix0= np.array(positions_matrix[j].split())[0:3]
		positions[j,:]  = positions_matrix0.astype(np.float)

		
		
	#numbers
	numbers = np.zeros(sum(atoms))
	counter=0
	atom_counter=1

	for ii in atoms:
		for kk in range(ii):
			numbers[counter]=atom_counter
			counter=counter+1
		atom_counter=atom_counter+1



	#seekpath
	structure = (cell,positions,numbers)	
	kpath_dictionary = seekpath.get_path(structure,with_time_reversal,recipe,threshhold,symprec,angle_tolerence)
	
	path_array = ['']*2*len(kpath_dictionary['path'])
	count=0
	count2=1
	for path_counter in kpath_dictionary['path']:
		path_array[count]=path_counter[0]
		path_array[count2]=path_counter[1]
		count=count+2
		count2=count2+2

	coord_matrix=np.zeros(shape=(2*len(kpath_dictionary['path']),3))
	path_array_counter = 0
	for mm in range(len(coord_matrix)):
		coord_matrix[mm,:] = np.dot(kpath_dictionary['point_coords'][path_array[path_array_counter]], supercell_matrix)
		path_array_counter=path_array_counter+1

	k_file = open('KPOINTS','w+')
	k_file.write("KPOINTS generated by PyProcar\n")
	k_file.write("%d ! Grid points\n"%grid_size)
	k_file.write("Line_mode\n")
	k_file.write("reciprocal\n")
	for iterator in range(len(coord_matrix)):
		if iterator%2==0:
			k_file.write('%f %f %f ! %s\n'% (coord_matrix[iterator,0],coord_matrix[iterator,1],coord_matrix[iterator,2],path_array[iterator])   )
		else:
			k_file.write('%f %f %f ! %s\n\n'% (coord_matrix[iterator,0],coord_matrix[iterator,1],coord_matrix[iterator,2],path_array[iterator])   )
	k_file.close()
