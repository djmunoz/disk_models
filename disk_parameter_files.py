#!/usr/bin/env python
#--------------------------------#
# disk_parameter_files.py
#--------------------------------#
""" 
Functions to generate AREPO/GADGET parameter files in consistency with initial conditions
created by the disks_3d_models module

Date created: 11/20/2016
"""


__author__= 'Diego J. Munoz'

import numpy as np


class paramfile():

    def __init__(self,*args,**kwargs):

        # Read parameters
        
        self.init_cond_file =  kwargs.get("init_cond_file")
        self.output_dir = kwargs.get("output_dir")
        self.snapshot_file_base = kwargs.get("snapshot_file_base")
        self.output_list_filename = kwargs.get("output_list_filename")

        self.ic_format = kwargs.get("ic_format")
        self.snap_format = kwargs.get("snap_format")

        self.time_limit_cpu = kwargs.get("time_limit_cpu")
        self.cpu_time_bet_restart_file = kwargs.get("cpu_time_bet_restart_file")
        self.resubmit_on = kwargs.get("resubmit_on")
        self.resubmit_command = kwargs.get("resubmit_command")

        self.max_mem_size = kwargs.get("max_mem_size")
        
        self.time_begin = kwargs.get("time_begin")
        self.time_max = kwargs.get("time_max")

        self.comoving_integration_on = kwargs.get("comoving_integration_on")
        self.periodic_boundaries_on = kwargs.get("periodic_boundaries_on")
        self.cooling_on = kwargs.get("cooling_on")
        self.star_formation_on = kwargs.get("star_formation_on")

        self.omega0 = kwargs.get("omega0")
        self.omega_lambda = kwargs.get("omega_lambda")
        self.omega_baryon = kwargs.get("omega_baryon")
        self.hubble_param = kwargs.get("hubble_param")
        self.box_size = kwargs.get("box_size")

        self.output_list_on = kwargs.get("output_list_on")
        self.time_bet_snapshot = kwargs.get("time_bet_snapshot")
        self.time_of_first_snapshot = kwargs.get("time_of_first_snapshot")
        self.time_bet_statistics = kwargs.get("time_bet_statistics")
        self.num_files_per_snapshot = kwargs.get("num_files_per_snapshot")
        self.num_files_written_in_parallel = kwargs.get("num_files_written_in_parallel")

        self.type_of_timestep_criterion = kwargs.get("type_of_timestep_criterion")
        self.err_tol_int_accuracy = kwargs.get("err_tol_int_accuracy")
        self.courant_fac = kwargs.get("courant_fac")
        self.max_size_timestep = kwargs.get("max_size_timestep")
        self.min_size_timestep = kwargs.get("min_size_timestep")
        
        self.init_gas_temp = kwargs.get("init_gas_temp")
        self.min_gas_temp = kwargs.get("min_gas_temp")
        self.minimum_density_on_startup = kwargs.get("minimum_density_on_startup")
        self.limit_u_below_this_density = kwargs.get("limit_u_below_this_density")
        self.limit_u_below_this_density_to_this_value = kwargs.get("limit_u_below_this_density_to_this_value")
        self.min_egy_spec = kwargs.get("min_egy_spec")

        self.type_of_opening_criterion = kwargs.get("type_of_opening_criterion")
        self.err_tol_theta = kwargs.get("err_tol_theta")
        self.err_tol_force_acc = kwargs.get("err_tol_force_acc")
        self.multiple_domains = kwargs.get("multiple_domains")
        self.top_node_factor = kwargs.get("top_node_factor")
        self.active_part_frac_for_new_domain_decomp = kwargs.get("active_part_frac_for_new_domain_decomp")

        self.des_num_ngb = kwargs.get("des_num_ngb")
        self.max_num_ngb_deviation = kwargs.get("max_num_ngb_deviation")

        self.unit_length_in_cm = kwargs.get("unit_length_in_cm")
        self.unit_mass_in_g = kwargs.get("unit_mass_in_g")
        self.unit_velocity_in_cm_per_s = kwargs.get("unit_velocity_in_cm_per_s")

        self.gravity_constant_internal                          = kwargs.get("gravity_constant_internal                         ")
        self.softening_comoving_type_0 = kwargs.get("softening_comoving_type_0")
        self.softening_comoving_type_1 = kwargs.get("softening_comoving_type_1")
        self.softening_comoving_type_2 = kwargs.get("softening_comoving_type_2")
        self.softening_comoving_type_3 = kwargs.get("softening_comoving_type_3")
        self.softening_comoving_type_4 = kwargs.get("softening_comoving_type_4")
        self.softening_comoving_type_5 = kwargs.get("softening_comoving_type_5")
        self.softening_max_phys_type_0 = kwargs.get("softening_max_phys_type_0")
        self.softening_max_phys_type_1 = kwargs.get("softening_max_phys_type_1")
        self.softening_max_phys_type_2 = kwargs.get("softening_max_phys_type_2")
        self.softening_max_phys_type_3 = kwargs.get("softening_max_phys_type_3")
        self.softening_max_phys_type_4 = kwargs.get("softening_max_phys_type_4")
        self.softening_max_phys_type_5 = kwargs.get("softening_max_phys_type_5")
        self.softening_type_of_parttype_0 = kwargs.get("softening_type_of_parttype_0")
        self.softening_type_of_parttype_1 = kwargs.get("softening_type_of_parttype_1")
        self.softening_type_of_parttype_2 = kwargs.get("softening_type_of_parttype_2")
        self.softening_type_of_parttype_3 = kwargs.get("softening_type_of_parttype_3")
        self.softening_type_of_parttype_4 = kwargs.get("softening_type_of_parttype_4")
        self.softening_type_of_parttype_5 = kwargs.get("softening_type_of_parttype_5")

        self.gas_soft_factor = kwargs.get("gas_soft_factor")
        self.cell_shaping_speed = kwargs.get("cell_shaping_speed")
        self.cell_max_angle_factor = kwargs.get("cell_max_angle_factor")

        self.mean_volume = kwargs.get("mean_volume")
        self.reference_gas_part_mass = kwargs.get("reference_gas_part_mass")
        self.target_gas_mass_factor = kwargs.get("target_gas_mass_factor")
        self.refinement_criterion = kwargs.get("refinement_criterion")
        self.derefinement_criterion = kwargs.get("derefinement_criterion")
        self.max_volume_diff = kwargs.get("max_volume_diff")
        self.min_volume = kwargs.get("min_volume")
        self.max_volume = kwargs.get("max_volume")


        # Set defaults ###############

        if (self.init_cond_file is None):
            self.init_cond_file = 'ics.dat'
        if (self.output_dir is None):
            self.output_dir = './output/'
        if (self.snapshot_file_base is None):
            self.snapshot_file_base = 'snap'
        if (self.output_list_filename is None):
            self.output_list_filename = 'foo'

        if (self.ic_format is None):
            self.ic_format = 3
        if (self.snap_format is None):
            self.snap_format = 3 

        if (self.time_limit_cpu is None):
            self.time_limit_cpu = 190000
        if (self.cpu_time_bet_restart_file is None):
            self.cpu_time_bet_restart_file = 7200
        if (self.resubmit_on is None):
            self.resubmit_on = 0
        if (self.resubmit_command is None):
            self.resubmit_command = 'my-scriptfile'
            
        if (self.max_mem_size is None):
            self.max_mem_size = 2300
        
        if (self.time_begin is None):
            self.time_begin = 0
        if (self.time_max is None):
            self.time_max = 10

        if (self.comoving_integration_on is None):
            self.comoving_integration_on = 0
        if (self.periodic_boundaries_on is None):
            self.periodic_boundaries_on = 0
        if (self.cooling_on is None):
            self.cooling_on = 0
        if (self.star_formation_on is None):
            self.star_formation_on = 0

        if (self.omega0 is None):
            self.omega0 = 0
        if (self.omega_lambda is None):
            self.omega_lambda = 0
        if (self.omega_baryon is None):
            self.omega_baryon = 0
        if (self.hubble_param is None):
            self.hubble_param = 0
        if (self.box_size is None):
            self.box_size = 1.0

        if (self.output_list_on is None):
            self.output_list_on = 0
        if (self.time_bet_snapshot is None):
            self.time_bet_snapshot = 1.0
        if (self.time_of_first_snapshot is None):
            self.time_of_first_snapshot = 0.0
        if (self.time_bet_statistics is None):
            self.time_bet_statistics = 0.1
        if (self.num_files_per_snapshot is None):
            self.num_files_per_snapshot = 1  
        if (self.num_files_written_in_parallel is None):
            self.num_files_written_in_parallel = 1

        if (self.type_of_timestep_criterion is None):
            self.type_of_timestep_criterion = 0
        if (self.err_tol_int_accuracy is None):
            self.err_tol_int_accuracy = 0.012
        if (self.courant_fac is None):
            self.courant_fac = 0.3
        if (self.max_size_timestep is None):
            self.max_size_timestep = 0.05
        if (self.min_size_timestep is None):
            self.min_size_timestep = 0
        
        if (self.init_gas_temp is None):
            self.init_gas_temp = 0
        if (self.min_gas_temp is None):
            self.min_gas_temp = 0
        if (self.minimum_density_on_startup is None):
            self.minimum_density_on_startup = 0
        if (self.limit_u_below_this_density is None):
            self.limit_u_below_this_density = 0
        if (self.limit_u_below_this_density_to_this_value is None):
            self.limit_u_below_this_density_to_this_value = 0
        if (self.min_egy_spec is None):
            self.min_egy_spec = 0

        if (self.type_of_opening_criterion is None):
            self.type_of_opening_criterion = 1
        if (self.err_tol_theta is None):
            self.err_tol_theta = 0.7
        if (self.err_tol_force_acc is None):
            self.err_tol_force_acc = 0.0025
        if (self.multiple_domains is None):
            self.multiple_domains = 1
        if (self.top_node_factor is None):
            self.top_node_factor = 1
        if (self.active_part_frac_for_new_domain_decomp is None):
            self.active_part_frac_for_new_domain_decomp = 0.5

        if (self.des_num_ngb is None):
            self.des_num_ngb  = 64
        if (self.max_num_ngb_deviation is None):
            self.max_num_ngb_deviation = 1

        if (self.unit_length_in_cm is None):
            self.unit_length_in_cm = 1
        if (self.unit_mass_in_g is None):
            self.unit_mass_in_g = 1
        if (self.unit_velocity_in_cm_per_s is None):
            self.unit_velocity_in_cm_per_s = 1

        if (self.gravity_constant_internal is None):
            self.gravity_constant_internal = 1
        if (self.softening_comoving_type_0 is None):
            self.softening_comoving_type_0 = 0.001
        if (self.softening_comoving_type_1 is None):
            self.softening_comoving_type_1 = 0.001
        if (self.softening_comoving_type_2 is None):
            self.softening_comoving_type_2 = 0.007
        if (self.softening_comoving_type_3 is None):
            self.softening_comoving_type_3 = 0.081
        if (self.softening_comoving_type_4 is None):
            self.softening_comoving_type_4 = 0.001
        if (self.softening_comoving_type_5 is None):
            self.softening_comoving_type_5 = 0.001
        if (self.softening_max_phys_type_0 is None):
            self.softening_max_phys_type_0 = 0.0005
        if (self.softening_max_phys_type_1 is None):
            self.softening_max_phys_type_1 = 0.0005
        if (self.softening_max_phys_type_2 is None):
            self.softening_max_phys_type_2 = 0.007
        if (self.softening_max_phys_type_3 is None):
            self.softening_max_phys_type_3 = 0.081
        if (self.softening_max_phys_type_4 is None):
            self.softening_max_phys_type_4 = 0.0005
        if (self.softening_max_phys_type_5 is None):
            self.softening_max_phys_type_5 = 0.0005
        if (self.softening_type_of_parttype_0 is None):
            self.softening_type_of_parttype_0 = 0
        if (self.softening_type_of_parttype_1 is None):
            self.softening_type_of_parttype_1 = 0
        if (self.softening_type_of_parttype_2 is None):
            self.softening_type_of_parttype_2 = 0
        if (self.softening_type_of_parttype_3 is None):
            self.softening_type_of_parttype_3 = 0
        if (self.softening_type_of_parttype_4 is None):
            self.softening_type_of_parttype_4 = 0
        if (self.softening_type_of_parttype_5 is None):
            self.softening_type_of_parttype_5 = 0

        if ( self.gas_soft_factor is None):
            self.gas_soft_factor = 2.5
        if (self.cell_shaping_speed is None):
            self.cell_shaping_speed = 0.5
        if (self.cell_max_angle_factor is None):
            self.cell_max_angle_factor = 1.4

        if (self.mean_volume is None):
            self.mean_volume = 1.0
        if (self.reference_gas_part_mass is None):
            self.reference_gas_part_mass = 1.0
        if (self.target_gas_mass_factor is None):
            self.target_gas_mass_factor = 1.0
        if (self.refinement_criterion is None):
            self.refinement_criterion = 1
        if (self.derefinement_criterion is None):
            self.derefinement_criterion = 1
        if (self.max_volume_diff is None):
            self.max_volume_diff = 10
        if (self.min_volume is None):
            self.min_volume = 0.001
        if (self.max_volume is None):
            self.max_volume = 1.0

        

    def write(self,filename):

        f = open(filename,'w')
        f.write("\n%----  Relevant files\n")
        f.write("InitCondFile\t\t %s\n" % self.init_cond_file)
        f.write("OutputDir\t\t %s\n" % self.output_dir)
        f.write("SnapshotFileBase\t %s\n" %  self.snapshot_file_base)
        f.write("OutputListFilename\t %s\n" % self.output_list_filename)

        f.write("\n%---- File formats\n")
        f.write("ICFormat\t\t %s\n" %  self.ic_format)                               
        f.write("SnapFormat\t\t %s\n" %  self.snap_format)

        f.write("\n%---- CPU-time limits\n")
        f.write("TimeLimitCPU\t\t %s\n" % self.time_limit_cpu)                                    
        f.write("CpuTimeBetRestartFile\t %s\n" % self.cpu_time_bet_restart_file)                           
        f.write("ResubmitOn\t\t %s\n" %  self.resubmit_on)                                     
        f.write("ResubmitCommand\t\t %s\n" % self.resubmit_command)

        f.write("\n%----- Memory alloction\n")
        f.write("MaxMemSize\t\t %s\n" % self.max_mem_size)

        f.write("\n%---- Caracteristics of run\n")
        f.write("TimeBegin\t\t %s\n" %  self.time_begin)                                      
        f.write("TimeMax\t\t\t %s\n" %  self.time_max)

        f.write("\n%---- Basic code options that set the type of simulation\n")
        f.write("ComovingIntegrationOn\t %s\n" % self.comoving_integration_on)                           
        f.write("PeriodicBoundariesOn\t %s\n" % self.periodic_boundaries_on)                            
        f.write("CoolingOn\t\t %s\n" % self.cooling_on)                                       
        f.write("StarformationOn\t\t %s\n" % self.star_formation_on)

        f.write("\n%---- Cosmological parameters\n")
        f.write("Omega0\t\t %s\n" %  self.omega0)                                         
        f.write("OmegaLambda\t %s\n" % self.omega_lambda)                                     
        f.write("OmegaBaryon\t %s\n" % self.omega_baryon)                                     
        f.write("HubbleParam\t %s\n" % self.hubble_param)                                     
        f.write("BoxSize\t\t %5.1f\n" % self.box_size)

        f.write("\n%---- Output frequency and output paramaters\n")
        f.write("OutputListOn\t\t\t %s\n" % self.output_list_on)                                    
        f.write("TimeBetSnapshot\t\t\t %s\n" % self.time_bet_snapshot)                                
        f.write("TimeOfFirstSnapshot\t\t %s\n" %  self.time_of_first_snapshot)                            
        f.write("TimeBetStatistics\t\t %s\n" % self.time_bet_statistics)                               
        f.write("NumFilesPerSnapshot\t\t %s\n" % self.num_files_per_snapshot)                             
        f.write("NumFilesWrittenInParallel\t %s\n" % self.num_files_written_in_parallel)

        f.write("\n%---- Accuracy of time integration\n")
        f.write("TypeOfTimestepCriterion\t\t %s\n" % self.type_of_timestep_criterion)                         
        f.write("ErrTolIntAccuracy\t\t %s\n" %  self.err_tol_int_accuracy)                              
        f.write("CourantFac\t\t\t %s\n" % self.courant_fac)                                      
        f.write("MaxSizeTimestep\t\t\t %s\n" % self.max_size_timestep)                                 
        f.write("MinSizeTimestep\t\t\t %s\n" % self.min_size_timestep)

        f.write("\n%---- Treatment of empty space and temperature limits\n")
        f.write("InitGasTemp\t\t\t %s\n" % self.init_gas_temp)                                     
        f.write("MinGasTemp\t\t\t %s\n" % self.min_gas_temp)                                      
        f.write("MinimumDensityOnStartUp\t\t %s\n" % self.minimum_density_on_startup)                         
        f.write("LimitUBelowThisDensity\t\t %s\n" % self.limit_u_below_this_density)                          
        f.write("LimitUBelowCertainDensityToThisValue\t %s\n" % self.limit_u_below_this_density_to_this_value)            
        f.write("MinEgySpec\t\t\t %s\n" % self.min_egy_spec)

        f.write("\n%---- Tree algorithm, force accuracy, domain update frequency\n")
        f.write("TypeOfOpeningCriterion\t\t\t %s\n" % self.type_of_opening_criterion)                          
        f.write("ErrTolTheta\t\t\t\t %s\n" % self.err_tol_theta)                                     
        f.write("ErrTolForceAcc\t\t\t\t %s\n" % self.err_tol_force_acc)                                  
        f.write("MultipleDomains\t\t\t\t %s\n" % self.multiple_domains)                                  
        f.write("TopNodeFactor\t\t\t\t %s\n" % self.top_node_factor)                                   
        f.write("ActivePartFracForNewDomainDecomp\t %s\n" % self.active_part_frac_for_new_domain_decomp)

        f.write("\n%---- Initial density estimate\n")
        f.write("DesNumNgb\t\t %s\n" %  self.des_num_ngb)                                      
        f.write("MaxNumNgbDeviation\t %s\n" % self.max_num_ngb_deviation)

        f.write("\n%---- System of units\n")
        f.write("UnitLength_in_cm\t\t %s\n" %  self.unit_length_in_cm)                              
        f.write("UnitMass_in_g\t\t\t %s\n" % self.unit_mass_in_g)
        f.write("UnitVelocity_in_cm_per_s\t %s\n" % self.unit_velocity_in_cm_per_s)                       

        f.write("\n%---- Gravitational softening length\n")
        f.write("GravityConstantInternal\t\t %s\n" % self.gravity_constant_internal)                         
        f.write("SofteningComovingType0\t\t %s\n" % self.softening_comoving_type_0)                          
        f.write("SofteningComovingType1\t\t %s\n" % self.softening_comoving_type_1)                                                    
        f.write("SofteningComovingType2\t\t %s\n" % self.softening_comoving_type_2)
        f.write("SofteningComovingType3\t\t %s\n" % self.softening_comoving_type_3)
        f.write("SofteningComovingType4\t\t %s\n" % self.softening_comoving_type_4)                          
        f.write("SofteningComovingType5\t\t %s\n" % self.softening_comoving_type_5)                           
        f.write("SofteningMaxPhysType0\t\t %s\n" % self.softening_max_phys_type_0)                           
        f.write("SofteningMaxPhysType1\t\t %s\n" % self.softening_max_phys_type_1)                            
        f.write("SofteningMaxPhysType2\t\t %s\n" % self.softening_max_phys_type_2)                            
        f.write("SofteningMaxPhysType3\t\t %s\n" % self.softening_max_phys_type_3)                            
        f.write("SofteningMaxPhysType4\t\t %s\n" % self.softening_max_phys_type_4)                            
        f.write("SofteningMaxPhysType5\t\t %s\n" % self.softening_max_phys_type_5)                            
        f.write("SofteningTypeOfPartType0\t\t %s\n" % self.softening_type_of_parttype_0)                        
        f.write("SofteningTypeOfPartType1\t\t %s\n" % self.softening_type_of_parttype_1)                         
        f.write("SofteningTypeOfPartType2\t\t %s\n" % self.softening_type_of_parttype_2)                         
        f.write("SofteningTypeOfPartType3\t\t %s\n" % self.softening_type_of_parttype_3)                         
        f.write("SofteningTypeOfPartType4\t\t %s\n" % self.softening_type_of_parttype_4)                         
        f.write("SofteningTypeOfPartType5\t\t %s\n" % self.softening_type_of_parttype_5)

        
        f.write("\n%----- Mesh regularization options\n")
        f.write("GasSoftFactor\t\t %s\n" % self.gas_soft_factor)                                   
        f.write("CellShapingSpeed\t %s\n" % self.cell_shaping_speed)                                
        f.write("CellMaxAngleFactor\t %s\n" % self.cell_max_angle_factor)                              

        f.write("\n%---- Refinement options\n")
        f.write("MeanVolume\t\t %s\n" % self.mean_volume)
        f.write("ReferenceGasPartMass\t %8.4e\n" % self.reference_gas_part_mass)
        f.write("TargetGasMassFactor\t %s\n" % self.target_gas_mass_factor)
        f.write("RefinementCriterion\t %s\n" % self.refinement_criterion)
        f.write("DerefinementCriterion\t %s\n" % self.derefinement_criterion)
        f.write("MaxVolumeDiff\t\t %f\n" % self.max_volume_diff)
        f.write("MinVolume\t\t %6.2e\n\n" % self.min_volume)
        f.write("MaxVolume\t\t %6.2e\n\n" % self.max_volume)

        f.close()
