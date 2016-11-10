# disk_3d_models
Sets of routines to setup three-dimensional models of accretion disks


To run these scripts and creat an HDF5 initial condition snapshot, follow the next steps:

1) Load module

> import disk_3d_models as d3d

2) Choose a density profile type (see 'disk_density_profiles.py' for already implemented axisymmetric profiles),e.g., a Lynden-Bell - Pringel self-similar profile modified to have a central cavity:

> sigma_type='similarity_cavity'

3) Create the disk model under this profile type (other parameters included)

> d = d3d.disk(sigma_type="similarity_cavity",csnd0=0.12,l=1.0,R_cav=2.5,xi=3.1,Rout=15,adiabatic_gamma=1.00001)

4)  Create the disk mesh data structure
>  mesh = d3d.disk_mesh(mesh_type="mc",Ncells=500000,fill_background=True, fill_center=True,fill_box=True,BoxSize=50)

5) Create an instance of the snapshot() structure, and generate a model snapshot
> s = d3d.snapshot()
> s.create(d,mesh)

6) Write the snapshot to a file
> s.write_snapshot(d,mesh,filename="disk.dat.hdf5",time=0)