# import readsnapHDF5 as rs
# header = rs.snapshot_header("snap_063.0") 
# mass = rs.read_block("snap_063","MASS",parttype=5) # reads mass for particles of type 5

import numpy as np
import os
import sys
import math
import h5py

############ 
#DATABLOCKS#
############
#descriptions of all datablocks -> add datablocks here!
#TAG:[HDF5_NAME,DIM]
datablocks = {"POS ":["Coordinates",3], 
	      "VEL ":["Velocities",3],
	      "ID  ":["ParticleIDs",1],
	      "MASS":["Masses",1],
              "U   ":["InternalEnergy",1],
              "RHO ":["Density",1],
              "VOL ":["Volume",1],
	      "PRES":["Pressure",1],
              "CMCE":["Center-of-Mass",3],
              "AREA":["Surface Area",1],
              "NFAC":["Number of faces of cell",1],
              "NE  ":["ElectronAbundance",1],
              "NH  ":["NeutralHydrogenAbundance",1],
	      "HSML":["SmoothingLength",1],
              "SFR ":["StarFormationRate",1],
              "AGE ":["StellarFormationTime",1],
              "Z   ":["Metallicity",1],
	      "ACCE":["Acceleration",3],
              "VEVE":["VertexVelocity",3],
              "FACA":["MaxFaceAngle",1],              
	      "COOR":["CoolingRate",1],
              "POT ":["Potential",1],
	      "MACH":["MachNumber",1],
	      "GAGE":["GFM StellarFormationTime",1],
              "GIMA":["GFM InitialMass",1],
              "GZ  ":["GFM Metallicity",1],
              "GMET":["GFM Metals",9],
              "GMRE":["GFM MetalsReleased",9],	      
              "GMAR":["GFM MetalMassReleased", 1],
	      "TSTP":["TimeStep", 1],
	      "SPIN":["CellSpin", 3],
	      "VORT":["Vorticity", 3],
	      "ANGM":["AngularMomentum", 3],
	      "GRAP":["PressureGradient", 3],
	      "GRAR":["DensityGradient", 3],
	      "GRAV":["VelocityGradient", 3],
	      "SECO":["SecondAreaMoment",4],
	      "TRNT":["NumTracers",1],
	      "TRID":["TracerID",1],
	      "TRPI":["ParentID",1]
	      }
#####################################################################################################################
#                                                    READING ROUTINES			                            #
#####################################################################################################################


########################### 
#CLASS FOR SNAPSHOT HEADER#
###########################  
class snapshot_header:
	def __init__(self, *args, **kwargs):
		if (len(args) == 1):
			filename = args[0]
 			if os.path.exists(filename):
			        curfilename=filename
			elif os.path.exists(filename+".hdf5"):
				curfilename = filename+".hdf5"
			elif os.path.exists(filename+".0.hdf5"): 
				curfilename = filename+".0.hdf5"
			else:	
				print "[error] file not found : ", filename
			    	sys.exit()
    
			f=h5py.File(curfilename)
                        header_dict = dict(f[f.keys()[f.keys().index('Header')]].attrs)
			self.npart = header_dict['NumPart_ThisFile'] 
			self.nall = header_dict['NumPart_Total']
			self.nall_highword = header_dict['NumPart_Total_HighWord']
			self.massarr = header_dict['MassTable'] 
			self.time = header_dict['Time'] 
			self.redshift = header_dict['Redshift'] 
			self.boxsize = header_dict['BoxSize']
			self.filenum = header_dict['NumFilesPerSnapshot']
			self.omega0 = header_dict['Omega0']
			self.omegaL = header_dict['OmegaLambda']
			self.hubble = header_dict['HubbleParam']
			self.sfr = header_dict['Flag_Sfr'] 
			self.cooling = header_dict['Flag_Cooling']
			self.double = header_dict['Flag_DoublePrecision']
			self.stellar_age = header_dict['Flag_StellarAge']
			self.metals = header_dict['Flag_Metals']
			self.feedback = header_dict['Flag_Feedback']
			f.close()

		else:
			#read arguments
			self.npart = kwargs.get("npart")
			self.nall = kwargs.get("nall")
			self.nall_highword = kwargs.get("nall_highword")
			self.massarr = kwargs.get("massarr")
			self.time = kwargs.get("time")
			self.redshift = kwargs.get("redshift")
			self.boxsize = kwargs.get("boxsize")
			self.filenum = kwargs.get("filenum")
			self.omega0 = kwargs.get("omega0")
			self.omegaL = kwargs.get("omegaL")
			self.hubble = kwargs.get("hubble")
			self.sfr = kwargs.get("sfr")
			self.cooling = kwargs.get("cooling")
			self.stellar_age = kwargs.get("stellar_age")
			self.metals = kwargs.get("metals")
			self.feedback = kwargs.get("feedback")
			self.double = kwargs.get("double")

			#set default values
                        if (self.npart is None):
				self.npart = np.array([0,0,0,0,0,0], dtype="int32")
                        if (self.nall is None):				
				self.nall  = np.array([0,0,0,0,0,0], dtype="uint32")
                        if (self.nall_highword is None):				
				self.nall_highword = np.array([0,0,0,0,0,0], dtype="uint32")
                        if (self.massarr is None):
				self.massarr = np.array([0,0,0,0,0,0], dtype="float64")
                        if (self.time is None):				
				self.time = np.array([0], dtype="float64")
                        if (self.redshift is None):				
				self.redshift = np.array([0], dtype="float64")
                        if (self.boxsize is None):				
				self.boxsize = np.array([0], dtype="float64")
                        if (self.filenum is None):
				self.filenum = np.array([1], dtype="int32")
                        if (self.omega0 is None):
				self.omega0 = np.array([0], dtype="float64")
                        if (self.omegaL is None):
				self.omegaL = np.array([0], dtype="float64")
                        if (self.hubble is None):
				self.hubble = np.array([0], dtype="float64")
                        if (self.sfr is None):	
				self.sfr = np.array([0], dtype="int32")            
                        if (self.cooling is None):	
				self.cooling = np.array([0], dtype="int32")
                        if (self.stellar_age is None):	
				self.stellar_age = np.array([0], dtype="int32")
                        if (self.metals is None):	
				self.metals = np.array([0], dtype="int32")
                        if (self.feedback is None):	
				self.feedback = np.array([0], dtype="int32")
                        if (self.double is None):	
				self.double = np.array([0], dtype="int32")
		
			
		

##############################
#READ ROUTINE FOR SINGLE FILE#
############################## 
def read_block_single_file(filename, block_name, dim2, parttype=-1, no_mass_replicate=False, fill_block_name="", verbose=False):

  if (verbose):
	  print "[single] reading file           : ", filename   	
	  print "[single] reading                : ", block_name
      
  head = snapshot_header(filename)
  npart = head.npart
  massarr = head.massarr
  nall = head.nall
  filenum = head.filenum
  doubleflag = head.double #GADGET-2 change
  #doubleflag = 0          #GADGET-2 change
  del head

  f=h5py.File(filename)


  #read specific particle type (parttype>=0, non-default)
  if parttype>=0:
        if (verbose):
        	print "[single] parttype               : ", parttype 
	if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0)):
	        if (verbose):
			print "[single] replicate mass block"	
		ret_val=np.repeat(massarr[parttype], npart[parttype])
        else:		
	  	part_name='PartType'+str(parttype)
	  	ret_val = f.__getitem__(part_name).__getitem__(block_name)[:]
        if (verbose):
        	print "[single] read particles (total) : ", ret_val.shape[0]/dim2

  #read all particle types (parttype=-1, default)
  if parttype==-1:
	first=True
	dim1=0
	for parttype in range(0,5):
		part_name='PartType'+str(parttype)
		if (f.keys().__contains__(part_name)):
			if (verbose):
				print "[single] parttype               : ", parttype 
				print "[single] massarr                : ", massarr
				print "[single] npart                  : ", npart


		        #replicate mass block per default (unless no_mass_replicate is set)
	        	if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0) & (no_mass_replicate==False)):
                       		if (verbose):
                               		print "[single] replicate mass block"
				if (first):
			        	data=np.repeat(massarr[parttype], npart[parttype])
					dim1+=data.shape[0]
					ret_val=data
					first=False
				else:	
					data=np.repeat(massarr[parttype], npart[parttype])	
                                        dim1+=data.shape[0]
                                        ret_val=np.append(ret_val, data)
        	                if (verbose):
                	        	print "[single] read particles (total) : ", ret_val.shape[0]/dim2
                                if (doubleflag==0):
					ret_val=ret_val.astype("float32")
                                else:
                                        ret_val=ret_val.astype("float64")

			#fill fill_block_name with zeros if fill_block_name is set and particle type is present and fill_block_name not already stored in file for that particle type
                        if ((block_name==fill_block_name) & (block_name!="Masses") & (npart[parttype]>0) & (f.__getitem__(part_name).__contains__(block_name)==False)):
                                if (verbose):
                                        print "[single] replicate block name", fill_block_name
                                if (first):
                                        data=np.repeat(0.0, npart[parttype]*dim2)
                                        dim1+=data.shape[0]
                                        ret_val=data
                                        first=False
                                else:
                                        data=np.repeat(0.0, npart[parttype]*dim2)
                                        dim1+=data.shape[0]
                                        ret_val=np.append(ret_val, data)
                                if (verbose):
                                        print "[single] read particles (total) : ", ret_val.shape[0]/dim2
                                if (doubleflag==0):
                                        ret_val=ret_val.astype("float32")
                                else:
                                        ret_val=ret_val.astype("float64")

			#default: just read the block
			if (f.__getitem__(part_name).__contains__(block_name)):
				if (first):
					data=f.__getitem__(part_name).__getitem__(block_name)[:]
					dim1+=data.shape[0]
					ret_val=data
					first=False
				else:
					data=f.__getitem__(part_name).__getitem__(block_name)[:]
					dim1+=data.shape[0]
					ret_val=np.append(ret_val, data)
                		if (verbose):
                        		print "[single] read particles (total) : ", ret_val.shape[0]/dim2

	if ((dim1>0) & (dim2>1)):
		ret_val=ret_val.reshape(dim1,dim2)

  f.close()

  return ret_val

##############
#READ ROUTINE#
##############
def read_block(filename, block, parttype=-1, no_mass_replicate=False, fill_block="", verbose=False):
  if (verbose):
          print "reading block          : ", block

  if parttype not in [-1,0,1,2,3,4,5,6]:
    print "[error] wrong parttype given"
    sys.exit()

  curfilename=filename+".hdf5"

  if os.path.exists(curfilename):
    multiple_files=False
  elif os.path.exists(filename+".0"+".hdf5"):
    curfilename = filename+".0"+".hdf5"
    multiple_files=True
  else:
    print "[error] file not found : ", filename
    sys.exit()

  head = snapshot_header(curfilename)
  filenum = head.filenum
  del head

 
  if (datablocks.has_key(block)):
        block_name=datablocks[block][0]
        dim2=datablocks[block][1]
        first=True
        if (verbose):
                print "Reading HDF5           : ", block_name
                print "Data dimension         : ", dim2
		print "Multiple file          : ", multiple_files
  else:
        print "[error] Block type ", block, "not known!"
        sys.exit()

  fill_block_name=""
  if (fill_block!=""):
        if (datablocks.has_key(fill_block)):
                fill_block_name=datablocks[fill_block][0]
                dim2=datablocks[fill_block][1]
                if (verbose):
                        print "Block filling active   : ", fill_block_name


  if (multiple_files):	
	first=True
	dim1=0
	for num in range(0,filenum):
		curfilename=filename+"."+str(num)+".hdf5"
		if (verbose):
			print "Reading file           : ", num, curfilename 
		if (first):
			data = read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)
			dim1+=data.shape[0]
			ret_val = data
			first = False 
		else:	 
                        data = read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)
                        dim1+=data.shape[0]
			ret_val=np.append(ret_val, data)
                if (verbose):
                        print "Read particles (total) : ", ret_val.shape[0]/dim2

	if ((dim1>0) & (dim2>1)):
		ret_val=ret_val.reshape(dim1,dim2)	
  else:
	ret_val=read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, fill_block_name, verbose)

  return ret_val


#############
#LIST BLOCKS#
#############
def list_blocks(filename, parttype=-1, verbose=False):
  
  f=h5py.File(filename)
  for parttype in range(0,5):
  	part_name='PartType'+str(parttype)
        if (f.__contains__(part_name)):
        	print "Parttype contains : ", parttype
		print "-------------------"
		iter = it=datablocks.__iter__()
		next = iter.next()
		while (1):
			if (verbose):
				print "check ", next, datablocks[next][0]
			if (f.__getitem__(part_name).__contains__(datablocks[next][0])):
  				print next, datablocks[next][0]
			try:
				next=iter.next()
			except StopIteration:
				break	
  f.close() 

#################
#CONTAINS BLOCKS#
#################
def contains_block(filename, tag, parttype=-1, verbose=False):
  
  contains_flag=False
  f=h5py.File(filename)
  for parttype in range(0,5):
        part_name='PartType'+str(parttype)
        if (f.__contains__(part_name)):
                iter = it=datablocks.__iter__()
                next = iter.next()
                while (1):
                        if (verbose):
                                print "check ", next, datablocks[next][0]
                        if (f.__getitem__(part_name).__contains__(datablocks[next][0])):
                                if (next.find(tag)>-1):
					contains_flag=True	
                        try:
                                next=iter.next()
                        except StopIteration:
                                break
  f.close() 
  return contains_flag

############
#CHECK FILE#
############
def check_file(filename):
  f=h5py.File(filename)
  f.close()
                                                                                                                                                  






#####################################################################################################################
#                                                    WRITING ROUTINES    		                            #
#####################################################################################################################



#######################
#OPEN FILE FOR WRITING#
#######################
def openfile(filename):
	f=h5py.File(filename, mode = "w")	 
	return f

############
#CLOSE FILE#
############
def closefile(f):
	f.close()

##############################
#WRITE SNAPSHOT HEADER OBJECT#
##############################
def writeheader(f, header):	
    	group_header=f.create_group("Header")
	group_header.attrs.__setitem__("NumPart_ThisFile",header.npart)
	group_header.attrs.__setitem__("NumPart_Total",header.nall)
	group_header.attrs.__setitem__("NumPart_Total_HighWord",header.nall_highword)
	group_header.attrs.__setitem__("MassTable",header.massarr)
	group_header.attrs.__setitem__("Time",header.time)
	group_header.attrs.__setitem__("Redshift",header.redshift)
	group_header.attrs.__setitem__("BoxSize",header.boxsize)
	group_header.attrs.__setitem__("NumFilesPerSnapshot",header.filenum)
        group_header.attrs.__setitem__("Omega0",header.omega0)
	group_header.attrs.__setitem__("OmegaLambda",header.omegaL)
        group_header.attrs.__setitem__("HubbleParam",header.hubble)	
	group_header.attrs.__setitem__("Flag_Sfr",header.sfr)	
	group_header.attrs.__setitem__("Flag_Cooling",header.cooling)
	group_header.attrs.__setitem__("Flag_StellarAge",header.stellar_age)
	group_header.attrs.__setitem__("Flag_Metals",header.metals)
	group_header.attrs.__setitem__("Flag_Feedback",header.feedback)
	group_header.attrs.__setitem__("Flag_DoublePrecision",header.double)
        
###############
#WRITE ROUTINE#
###############
def write_block(f, block, parttype, data):
	part_name="PartType"+str(parttype)
	if (f.__contains__(part_name)==False):
	    	group=f.create_group(part_name)
	else:
		group=f.__getitem__(part_name)	
	
	if (datablocks.has_key(block)):
        	block_name=datablocks[block][0]
	        dim2=datablocks[block][1]		
		if (group.__contains__(block_name)==False):
			table=group.create_dataset(block_name, data=data)
		else:
			print "I/O block already written"
	else:
		print "Unknown I/O block"		

	



