#!/usr/bin/env python
#--------------------------------#
# disk_parameter_files.py
#--------------------------------#
from __future__ import print_function
""" 
Functions to generate AREPO/GADGET parameter files in consistency with initial conditions
created by the disks_3d_models module

Date created: 11/20/2016
"""


__author__= 'Diego J. Munoz'

import numpy as np
from collections import OrderedDict

# Dictionary containing all possible parameters
parameters_basic = {
    "InitCondFile":"init_cond_file",
    "OutputDir":"output_dir",
    "SnapshotFileBase":"snapshot_file_base",
    "OutputListFilename":"output_list_filename",
    # File formats
    "ICFormat":"ic_format",
    "SnapFormat":"snap_format",
    # CPU options
    "TimeLimitCPU":"time_limit_cpu",
    "CpuTimeBetRestartFile":"cpu_time_bet_restart_file",
    "ResubmitOn":"resubmit_on",
    "ResubmitCommand":"resubmit_command",
    # Memory allocation
    "MaxMemSize":"max_mem_size",
    # Run time
    "TimeBegin":"time_begin",
    "TimeMax":"time_max",
    # Basic options
    "ComovingIntegrationOn":"comoving_integration_on",
    "PeriodicBoundariesOn":"periodic_boundaries_on",
    "CoolingOn":"cooling_on",
    "StarformationOn":"star_formation_on",
    # Cosmological parameters
    "Omega0":"omega0",
    "OmegaLambda":"omega_lambda",
    "OmegaBaryon":"omega_baryon",
    "HubbleParam":"hubble_param",
    "BoxSize":"box_size",
    # Output properties
    "OutputListOn":"output_list_on",
    "TimeBetSnapshot":"time_bet_snapshot",
    "TimeOfFirstSnapshot":"time_of_first_snapshot",
    "TimeBetStatistics":"time_bet_statistics",
    "NumFilesPerSnapshot":"num_files_per_snapshot",
    "NumFilesWrittenInParallel":"num_files_written_in_parallel",
    # Integration options
    "TypeOfTimestepCriterion":"type_of_timestep_criterion",
    "ErrTolIntAccuracy":"err_tol_int_accuracy",
    "CourantFac":"courant_fac",
    "MaxSizeTimestep":"max_size_timestep",
    "MinSizeTimestep":"min_size_timestep",
    # Background properties
    "InitGasTemp":"init_gas_temp",
    "MinGasTemp":"min_gas_temp",
    "MinimumDensityOnStartUp":"minimum_density_on_startup",
    "LimitUBelowThisDensity":"limit_u_below_this_density",
    "LimitUBelowCertainDensityToThisValue":"limit_u_below_certain_density_to_this_value",
    "MinEgySpec":"min_egy_spec",
    # Tree, domain decomposition
    "TypeOfOpeningCriterion":"type_of_opening_criterion",
    "ErrTolTheta":"err_tol_theta",
    "ErrTolForceAcc":"err_tol_force_acc",
    "MultipleDomains":"multiple_domains",
    "TopNodeFactor":"top_node_factor",
    "ActivePartFracForNewDomainDecomp":"active_part_frac_for_new_domain_decomp",
    # Initial density estimate
    "DesNumNgb":"des_num_ngb",
    "MaxNumNgbDeviation":"max_num_ngb_deviation",
    # Units
    "UnitLength_in_cm":"unit_length_in_cm",
    "UnitMass_in_g":"unit_mass_in_g",
    "UnitVelocity_in_cm_per_s":"unit_velocity_in_cm_per_s",
    # Gravitational softenings
    "GravityConstantInternal":"gravity_constant_internal",
    "SofteningComovingType0":"softening_comoving_type_0",
    "SofteningComovingType1":"softening_comoving_type_1",
    "SofteningComovingType2":"softening_comoving_type_2",
    "SofteningComovingType3":"softening_comoving_type_3",
    "SofteningComovingType4":"softening_comoving_type_4",
    "SofteningComovingType5":"softening_comoving_type_5",
    "SofteningMaxPhysType0":"softening_max_phys_type_0",
    "SofteningMaxPhysType1":"softening_max_phys_type_1",
    "SofteningMaxPhysType2":"softening_max_phys_type_2",
    "SofteningMaxPhysType3":"softening_max_phys_type_3",
    "SofteningMaxPhysType4":"softening_max_phys_type_4",
    "SofteningMaxPhysType5":"softening_max_phys_type_5",
    "SofteningTypeOfPartType0":"softening_type_of_part_type_0",
    "SofteningTypeOfPartType1":"softening_type_of_part_type_1",
    "SofteningTypeOfPartType2":"softening_type_of_part_type_2",
    "SofteningTypeOfPartType3":"softening_type_of_part_type_3",
    "SofteningTypeOfPartType4":"softening_type_of_part_type_4",
    "SofteningTypeOfPartType5":"softening_type_of_part_type_5",                  
    # Mesh options
    "GasSoftFactor":"gas_soft_factor",
    "CellShapingSpeed":"cell_shaping_speed",
    "CellMaxAngleFactor":"cell_max_angle_factor",
    # Refinement options
    "MeanVolume":"mean_volume",
    "ReferenceGasPartMass":"reference_gas_part_mass",
    "TargetGasMassFactor":"target_gas_mass_factor",
    "RefinementCriterion":"refinement_criterion",
    "DerefinementCriterion":"derefinement_criterion",
    "MaxVolumeDiff":"max_volume_diff",
    "MinVolume":"min_volume",
    "MaxVolume":"max_volume"
}

parameters_other = OrderedDict([
    # Circumstellar disk options
    ("IsoSoundSpeed","iso_sound_speed"),
    ("AlphaCoefficient","alpha_coefficient"),
    ("InnerRadius","inner_radius"),
    ("OuterRadius","outer_radius"),
    ("EvanescentBoundaryStrength","evanescent_boundary_strength"),
    ("CircumstellarBoundaryDensity","circumstellar_boundary_density"),
    ("MaxBackgroundVolume","max_background_volume"),
    ("MinBackgroundVolume","min_background_volume"),
    ("MinBackgroundMass","min_background_mass"),
    ("MeanVolume","mean_volume"),
    ("IgnoreRefinementsBeyondThisRadius","ignore_refinements_beyond_this_radius"),
    ("IgnoreRefinementsWithinThisRadius","ignore_refinements_within_this_radius"),
    ("CircumstellarSoundSpeedDefault","circumstellar_sound_speed_default"),
    ("CircumstellarTempProfileIndex","circumstellar_temp_profile_index"),
    # Boundaries options
    ("BoundaryLayerScaleFactor","boundary_layer_scale_factor"),
    ("SpecialBoundarySpeed","special_boundary_speed"),
    ("SpecialBoundaryMotion","special_boundary_motion"),
    ("SpecialBoundaryType","special_boundary_type"),
    ("OutflowPressure","outflow_pressure"),
    # Binary options
    ("BinaryMassRatio","binary_mass_ratio"),
    ("BinarySoftening","binary_softening"),
    ("BinaryGrowthTime","binary_growth_time"),
    ("BinaryEccentricity","binary_eccentricity"),
    ("BinaryBarycentricCoord","binary_barycentric_coord"),
    # Accretion options
    ("CircumstellarSinkRadius","circumstellar_sink_radius"), 
    ("CircumstellarSinkCriterion","circumstellar_sink_criterion"),
    ("CircumstellarSinkEfficiency","circumstellar_sink_efficiency"),
    # Central Potential options
    ("CentralMass", "central_mass"),
    ("CentralMassSoftening", "central_mass_softening"),
    ("CentralAccretionRadius","central_accretion_radius"),
    ("CentralAccretionMethod","central_accretion_method")
])



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

        # Optional parameters
        self.central_mass = kwargs.get("central_mass")
        self.softening_central_mass = kwargs.get("softening_central_mass")
        self.iso_sound_speed = kwargs.get("iso_sound_speed")
        
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
        f.write("BoxSize\t\t %s\n" % self.box_size)

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
        f.write("ReferenceGasPartMass\t %s\n" % self.reference_gas_part_mass)
        f.write("TargetGasMassFactor\t %s\n" % self.target_gas_mass_factor)
        f.write("RefinementCriterion\t %s\n" % self.refinement_criterion)
        f.write("DerefinementCriterion\t %s\n" % self.derefinement_criterion)
        f.write("MaxVolumeDiff\t\t %s\n" % self.max_volume_diff)
        f.write("MinVolume\t\t %s\n" % self.min_volume)
        f.write("MaxVolume\t\t %s\n" % self.max_volume)

        
        # Optional parameters
        f.write("\n%---- Other options\n")
        for paramname in parameters_other:
            attr= parameters_other[paramname]
            if attr in self.__dict__.keys():
                if (attr is not None):
                    if (getattr(self,attr) is not None):
                        #paramname= parameters_other.keys()[parameters_other.values().index(attr)]
                        paramname= list(parameters_other.keys())[list(parameters_other.values()).index(attr)]
                        if (len(paramname) > 25):
                            f.write("%s\t%s\n" % (paramname, getattr(self,attr)))
                        elif (len(paramname) > 15):
                            f.write("%s\t\t%s\n" % (paramname, getattr(self,attr)))
                        else:
                            f.write("%s\t\t\t%s\n" % (paramname, getattr(self,attr)))

        # Disk parameters


        # Boundary parameters

        # Binary parameters

            
        f.close()

        
    def read(self,filename):
        '''
        Method of the paramfile class to read in a plain text parameter
        file and load the parameters of the class.

        '''

        with open(filename) as f:
            for line in f:
                if ('%' in line[0]): continue
                if not line.strip(): continue
                paramname = line.split(' ')[0]
                if ('\t' in paramname): paramname=paramname.split('\t')[0]
                paramval = line[len(paramname):].strip()
                if (' ' in paramval): paramval = paramval.split(' ')[0]
                if (paramname in parameters_basic):
                    setattr(self, parameters_basic[paramname], paramval)
                if (paramname in parameters_other):
                    setattr(self, parameters_other[paramname], paramval)
                    
                '''
                # Files
                if "InitCondFile" in line:
                if "OutputDir" in line:
                if "SnapshotFileBase" in line:
                if "OutputListFilename" in line:

                # File formats
                if "ICFormat" in line:
                if "SnapFormat" in line:

                # CPU options
                if "TimeLimitCPU" in line:
                if "CpuTimeBetRestartFile" in line:
                if "ResubmitOn" in line:
                if "ResubmitCommand" in line:

                # Memory allocation
                if "MaxMemSize" in line:

                # Run time
                if "TimeBegin" in line:
                if "TimeMax" in line:

                # Basic options
                if "ComovingIntegrationOn" in line:
                if "PeriodicBoundariesOn" in line:
                if "CoolingOn" in line:
                if "StarformationOn" in line:

                # Cosmological parameters
                if "Omega0" in line:
                if "OmegaLambda" in line:
                if "OmegaBaryon" in line:
                if "HubbleParam" in line:
                if "BoxSize" in line:

                # Output properties
                if "OutputListOn" in line:
                if "TimeBetSnapshot" in line:
                if "TimeOfFirstSnapshot" in line:
                if "TimeBetStatistics" in line:
                if "NumFilesPerSnapshot" in line:
                if "NumFilesWrittenInParallel" in line:

                # Integration options
                if "TypeOfTimestepCriterion" in line:
                if "ErrTolIntAccuracy" in line:
                if "CourantFac" in line:
                if "MaxSizeTimestep" in line:
                if "MinSizeTimestep" in line:
                
                # Background properties
                if "InitGasTemp" in line:
                if "MinGasTemp" in line:
                if "MinimumDensityOnStartUp" in line:
                if "LimitUBelowThisDensity" in line:
                if "LimitUBelowCertainDensityToThisValue" in line:
                if "MinEgySpec" in line:

                # Tree, domain decomposition
                if "TypeOfOpeningCriterion" in line:
                if "ErrTolTheta" in line:
                if "ErrTolForceAcc" in line:
                if "MultipleDomains" in line:
                if "TopNodeFactor" in line:
                if "ActivePartFracForNewDomainDecomp" in line:

                # Initial density estimate
                if "DesNumNgb" in line:
                if "MaxNumNgbDeviation" in line:
                
                # Units
                if "UnitLength_in_cm" in line:
                if "UnitMass_in_g" in line:
                if "UnitVelocity_in_cm_per_s" in line:
                
                # Gravitational softenings
                if "GravityConstantInternal" in line:
                if "SofteningComovingType0" in line:
                if "SofteningComovingType1" in line:
                if "SofteningComovingType2" in line:
                if "SofteningComovingType3" in line:
                if "SofteningComovingType4" in line:
                if "SofteningComovingType5" in line:
                if "SofteningMaxPhysType0" in line:
                if "SofteningMaxPhysType1" in line:
                if "SofteningMaxPhysType2" in line:
                if "SofteningMaxPhysType3" in line:
                if "SofteningMaxPhysType4" in line:
                if "SofteningMaxPhysType5" in line:
                if "SofteningTypeOfPartType0" in line:
                if "SofteningTypeOfPartType1" in line:
                if "SofteningTypeOfPartType2" in line:
                if "SofteningTypeOfPartType3" in line:
                if "SofteningTypeOfPartType4" in line:
                if "SofteningTypeOfPartType5" in line:                    
        
                # Mesh options
                if "GasSoftFactor" in line:
                if "CellShapingSpeed" in line:
                if "CellMaxAngleFactor" in line:
                
                # Refinement options
                if "MeanVolume" in line:
                if "ReferenceGasPartMass" in line:
                if "TargetGasMassFactor" in line:
                if "RefinementCriterion" in line:
                if "DerefinementCriterion" in line:
                if "MaxVolumeDiff" in line:
                if "MinVolume" in line:
                if "MaxVolume" in line:



                # Circumstellar disk option

                # Boundary options

                # Binary options
                '''
