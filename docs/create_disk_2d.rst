Creating a disk model in 2D
------------------------------

First, we load the :code:`disk_models` module.


.. code:: python

          import disk_models as dm
	  import disk_data_analysis as dda

Note that we have also loaded a different package that will help us plot
the model we have created


Now we create the disk snapshot

.. code:: python
	  
	  sigma_type='powerlaw_cavity'
	  d = dm.disk2d(sigma_type=sigma_type,csnd0=0.05,p=0.5,l=1.0,R_cav=2.5,xi=3.1,Rout=70,adiabatic_gamma=1.0,boundary_out = True)
	  mesh = dm.disk_mesh2d(mesh_type="polar",Ncells=500000,fill_background=True, fill_center=True,fill_box=True,BoxSize=160)
	  
