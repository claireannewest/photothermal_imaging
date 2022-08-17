import numpy as np

nback=VAL # refractive index of background at room temperature
diel_path=VAL # location to store dielectric files
shapefile=VAL
JA = shapefile[:,0]
x_shape = shapefile[:,1]
y_shape = shapefile[:,2]
z_shape = shapefile[:,3]

### Locate the JA that cooresponds to the beginning and end of each particle ###
def count_particles():
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

### Calculate the average temperature of each particle (so we can assign them each a temp-dep dielectric file.)
def avgtemp_of_parts():
	tempfile = np.loadtxt('temp.out')
	x = tempfile[:,0]
	y = tempfile[:,1]
	z = tempfile[:,2]
	temp_raw = tempfile[:,3]
	temp_points = np.column_stack((x, y, z))

	shape_points = np.column_stack((x_shape, y_shape, z_shape))
	temp_points = np.column_stack((x, y, z, np.round(temp_raw)))

	startingrow_eachpart, lastrow_eachpart = count_particles()
	numParts = len(startingrow_eachpart)

	temp_of_eachpart = np.zeros((numParts))

	alltogether = np.zeros((len(x_shape), 5))
	alltogether[:,0] = JA
	alltogether[:,1] = x_shape
	alltogether[:,2] = y_shape
	alltogether[:,3] = z_shape
	count = 0
	for particle in range(0,numParts): #loop over the particles individually 
		row_start = startingrow_eachpart[particle]
		row_end = lastrow_eachpart[particle]
		temps = np.zeros(row_end-row_start+1)
		ICOMP = np.zeros(row_end-row_start+1)

		points_in_particle = np.column_stack((x_shape[row_start:row_end+1], y_shape[row_start:row_end+1], z_shape[row_start:row_end+1], temps, ICOMP))
		for eachpoint in range(0,len(x_shape[row_start:row_end+1])):
			tempidx = np.where((points_in_particle[eachpoint,0] == temp_points[:,0]) & (points_in_particle[eachpoint,1] == temp_points[:,1]) & (points_in_particle[eachpoint,2] == temp_points[:,2]))
			## Check to make sure I'm grabbing the right point
			#print particle, points_in_particle[eachpoint,0:3], temp_points[tempidx,:][0][0]
			points_in_particle[eachpoint,3] = int(temp_points[tempidx,3][0][0])
			points_in_particle[eachpoint,4] = int(eachpoint)
			alltogether[count,4] = particle+1
			count = count+1
		temp_of_eachpart[particle] = np.mean(points_in_particle[:,3])
	return alltogether, temp_of_eachpart

def write_files():
	alltogether, temp_of_eachpart = avgtemp_of_parts()
	startingrow_eachpart, lastrow_eachpart = count_particles()

	## Write shape file 
	file = open('shape.dat','w')
	file.write(str('Shape with each particle labeled') + '\n')	
	file.write('\t' + str(int(JA[-1])) + ' = number of dipoles in target' + '\n')
	file.write(' 1.000000 0.000000 0.000000 = A_1 vector' + '\n')
	file.write(' 0.000000 1.000000 0.000000 = A_2 vector' + '\n')
	file.write(' 1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d' + '\n')
	file.write(' 0.000000 0.000000 0.000000 = (x,y,z)/d' + '\n')
	file.write(' JA  IX  IY  IZ ICOMP(x,y,z)'+ '\n')
	for i in range(0, len(JA)):
	    file.write( str(np.int(alltogether[i,0])) + '\t' + str(np.int(alltogether[i,1])) + '\t' + str(np.int(alltogether[i,2])) + '\t' + str(np.int(alltogether[i,3])) + '\t' + str(np.int(alltogether[i,4])) + '\t' + str(np.int(alltogether[i,4])) + '\t' + str(np.int(alltogether[i,4])) + '\n')
	file.close()

	## Write ddscat_filler file
	numParts = len(startingrow_eachpart)
	file = open('ddscat_filler','w')
	for i in range(0, numParts):
	        file.write(' ' + str('"') + str(diel_path) + str('au_Conor_') + str(np.int(temp_of_eachpart[i])) + str('K.txt"') + str(' = file with metal refractive index \
	') + '\n')
	file.close()

	## Avg. temps and write n_back 
	dTmax = np.mean(temp_of_eachpart)
	n_T = nback + 2.7e-4*dTmax
	file = open('n_T_of_temp_max','w')
	file.write(str(n_T))
	file.close()

write_files()

