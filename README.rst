**************************************************************************************
disk_models - Sets of routines to setup three-dimensional models of accretion disks
**************************************************************************************

.. class:: no-web
           
   .. image:: example_figures/mesh.png


Overview
--------


Sets of routines to setup three-dimensional models of accretion disks in hydrostatic and centrifugal equilibrium.

Installation
------------

You need to have git installed. In addition, you need the NumPy, SciPy and PyTables Python packages.

.. code::
   
   git clone https://github.com/djmunoz/disk_models.git

   cd disk_models
   
   sudo python setup.py install

If you do not have root permission, you might have to do

.. code::
   
   python setup.py install --user

To install the package in your :code:`$HOME/.local/`
   
That is all!


.. class:: no-web
           
   .. image:: example_figures/fragmentation.png


Basic usage
-----------

To run these scripts and creat an HDF5 initial condition snapshot, follow the next steps:

a.     **Load module:**
   
.. code:: python

	  import disk_models as dm


b. **Choose a density profile type:**
   Choose a density profile type (see 'disk_density_profiles.py' for already implemented axisymmetric profiles),e.g., a Lynden-Bell - Pringel self-similar profile modified to have a central cavity:
   
.. code:: python
	  
	  sigma_type='similarity_cavity'

c. **Create disk model:**
   Create the disk model under this profile type (other parameters included)

.. code:: python

	  d = dm.disk3d(sigma_type="similarity_cavity",csnd0=0.12,l=1.0,R_cav=2.5,xi=3.1,Rout=15,adiabatic_gamma=1.00001)

d. **Create mesh:**
   Create the disk mesh data structure
   
.. code:: python

	  mesh = dm.disk_mesh3d(mesh_type="mc",Ncells=500000,fill_background=True, fill_center=True,fill_box=True,BoxSize=50)

e. **Create the initial snapshot:**
   Create an instance of the snapshot() structure, and generate a model snapshot
   
.. code:: python
	  
	  s = dm.snapshot()
	  s.create(d,mesh)
	  
f. **Save to disk:**
   Write the snapshot to a file
   

.. code:: python
	  
	  s.write_snapshot(d,mesh,filename="disk.dat.hdf5",time=0)

