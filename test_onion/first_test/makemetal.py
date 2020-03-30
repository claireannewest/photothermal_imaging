import numpy as np
import matplotlib.pyplot as plt

nback = 1.0 #refractive index of background at room temperature
diel_path = '/gscratch/chem/clairew4/dielectrics/' #location to store dielectric files
cutoff = 2.

#######################################################################################

shapefile = np.loadtxt('shape.dat_test',skiprows=7)
JA = shapefile[:,0]
x_shape = shapefile[:,1]
y_shape = shapefile[:,2]
z_shape = shapefile[:,3]
shape_points = np.column_stack((x_shape, y_shape, z_shape))

tempfile = np.loadtxt('temp.out_test')
x = tempfile[:,0]
y = tempfile[:,1]
z = tempfile[:,2]
temp_round = np.round(tempfile[:,3])
temp_points = np.column_stack((x, y, z, temp_round))

#######################################################################################
#######################################################################################

def count_particles():
	### Locate the JA that cooresponds to the beginning and end of each particle ###
	startingrow_eachpart = np.array([int(0)]) #Row each new particle. It'll always start with 0.
	for i in range(0,len(x_shape)-1): 
		# Loop that counts the particles; relies on jumps in the x axis. 
		# I have to substract one because my loop adds one in the indexing step
		if abs(x_shape[i] - x_shape[i+1]) > 1.0:
			startingrow_eachpart = np.append(startingrow_eachpart, i+1)
			#print y_shape[i], z_shape[i]

	numParts = len(startingrow_eachpart)
	lastrow_eachpart = np.zeros(numParts,dtype=int) #JA of end of each new particle. It'll always be same length as firstrow_eachpart
	for i in range(0,numParts-1):
		if numParts != 1: 
			lastrow_eachpart[i] = int(startingrow_eachpart[i+1]-1)
	lastrow_eachpart[-1] = int(len(x_shape)-1)
	return startingrow_eachpart, lastrow_eachpart

def array_together():
	### define array alltogether, which will be used to write the final shape file
	startingrow_eachpart, lastrow_eachpart = count_particles()
	numParts = len(startingrow_eachpart)
	temp_of_eachpart = np.zeros((numParts))
	JA = np.zeros(( tempfile.shape[0] ))
	ICOMP = np.zeros(( tempfile.shape[0] ))
	alltogether = np.column_stack(( x, y, z, ICOMP ))
	return numParts, startingrow_eachpart, lastrow_eachpart

writeit = np.zeros((len(x), 4))

def label_particles():
	numParts, startingrow_eachpart, lastrow_eachpart = array_together()
	count = 0
	temp_metal = np.zeros(numParts) # each position is the average temperature of each particle 

	for particle in range(0, numParts): 
		### loop over the particles individually and assign ICOMP
		row_start = startingrow_eachpart[particle]
		row_end = lastrow_eachpart[particle]
		points_in_particle = np.column_stack((x_shape[row_start:row_end+1], y_shape[row_start:row_end+1], z_shape[row_start:row_end+1]))
		for line in points_in_particle:
			# now, we'll loop through every coordinate in this given particle, and find where it is contained in temp.out
			idx = np.where((temp_points[:,0:3] == line).all(axis=1))
			# next, put that temperature into the correct position in array temp_metal that corresponds to that particle, 
			# then add up each entry so all temp points can be averaged.
			temp_metal[ particle ] = temp_metal[ particle ] + temp_points[idx,3][0][0]
			writeit[count,:] = np.array([line[0], line[1], line[2], particle+1])
			count = count + 1
		temp_metal[particle] = temp_metal[particle]/len(points_in_particle)
	return count, numParts, temp_metal, writeit

def label_background():
	count, numParts, temp_metal, writeit = label_particles()
	for line in temp_points:
		idx = np.where((shape_points == line[0:3]).all(axis=1))
		if idx[0].shape == (0,) and line[3] > cutoff:
			### current coordinate, line, is a background point because it's not a shape point, 
			### and the temperature is greater than the cutoff, so write it!
			ICOMP = max(temp_round) - line[3] + numParts+1
			writeit[count,:] = np.array([line[0], line[1], line[2], ICOMP])
			count = count+1
	return count, temp_metal, writeit

def final_writeit():
	count, temp_metal, writeit = label_background()
	cleanup_writeit = writeit[0:count,:]
	JA = np.linspace(1, count+1, count+1)

	## Write shape file ###
	file = open('shape.dat_update','w')
	file.write(str('Shape with heated indices') + '\n')	
	file.write('\t' + str(int(count)) + ' = number of dipoles in xtarget' + '\n')
	file.write(' 1.000000 0.000000 0.000000 = A_1 vector' + '\n')
	file.write(' 0.000000 1.000000 0.000000 = A_2 vector' + '\n')
	file.write(' 1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d' + '\n')
	file.write(' 0.000000 0.000000 0.000000 = (x,y,z)/d' + '\n')
	file.write(' JA  IX  IY  IZ ICOMP(x,y,z)'+ '\n')
	for i in range(0, count):
	     file.write( str(np.int(JA[i])) + '\t' + str(np.int(cleanup_writeit[i,0])) + '\t' + str(np.int(cleanup_writeit[i,1])) + '\t' + str(np.int(cleanup_writeit[i,2])) + '\t' + str(np.int(cleanup_writeit[i,3])) + '\t' + str(np.int(cleanup_writeit[i,3])) + '\t' + str(np.int(cleanup_writeit[i,3])) + '\n')
	file.close()

	## Write ddscat_filler file ###
	numParts = len(temp_metal)
	file = open('ddscat_filler','w')
	for i in range(0, numParts):
		file.write(' ' + str('"') + str(diel_path) + str('au_Conor_') + str(np.int(temp_metal[i])) + str('K.txt"') + str(' = file with heated metal refractive index ') + '\n')
	
	ICOMP = cleanup_writeit[:,3]
	glyc_idxICOMPS = np.where(ICOMP > numParts)
	glyc_ICOMPs = ICOMP[glyc_idxICOMPS]
	unique_glyc_ICOMPs = np.unique(glyc_ICOMPs)

	for i in range(0, len(unique_glyc_ICOMPs)):
		ICOMP = unique_glyc_ICOMPs[i]
		temp = max(temp_round) - ICOMP + numParts+1
		file.write(' ' + str('"') + str(diel_path) + str('glyc_') + str(np.int(temp)) + str('K.txt"') + str(' = file with heated glycerol refractive index ') + '\n')
	file.close()

final_writeit()

def make_diel_files():
	T_r_un = range(0, 100)
	for i in range(0,len(T_r_un)):
	    dT = T_r_un[i]
	    n_T = nback + 2.7e-4*dT
	    w = np.linspace(0.5,4,101)
	    n = np.linspace(n_T, n_T, 101)
	    k = 0*w
	    Data = np.array([1.24/w[::-1], n[::-1], k[::-1]]).T
	    file = open(str(diel_path) + str('glyc_') + str(np.int(dT)) + str('K.txt'),'w')
	    file.write(str('nT of heated glycerol') + '\n')
	    file.write(str('1 2 3 0 0 = Specifies n or Eps') + '\n')
	    file.write(str('lambda	n 	k') + '\n')
	    for j in range(0, len(Data)):
	    	file.write(str(Data[j,0]) + '\t' + str(Data[j,1]) + '\t' + str(Data[j,2]) + '\n')
	    file.close()


make_diel_files()
