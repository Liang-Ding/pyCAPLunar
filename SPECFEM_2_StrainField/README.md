![instruction](https://github.com/Liang-Ding/pyCAPLunar/blob/master/SPECFEM_2_StrainField/SPECFEM3D_Cartesian_2_strain_field.png)

The following steps are to build the SGT database using the SPECFEM_Cartesian package.

The current version of the SPECFEM_Cartesian package used in this package: v2.0.2
The later versions of the SPECFEM_Cartesian package might work well.

(1) Change the following scripts in the SPECFEM_Cartesian package.  
.
├── dl_runtime_saver.F90		# Add to the folder */specfem3d/src/specfem3D/
├── iterate_time.F90			# Replace the original file in the folder */specfem3d/src/specfem3D/iterate_time.F90
└── rules.mk					# Replace the original file in the folder */specfem3d/src/specfem3D/rules.mk


(2) make the *.bin files. 
Please refer to the SPECFEM_Cartesian tutorial. 


(3) Make sure the following parameters are all set in the Par_file:
* SIMULATION_TYPE =  1
* SAVE_FORWARD 	  = .true. 
* ATTENUATION  	  = .true.


(4) Run the waveform simulation that will automatically store the strain Green's tensor (SGT) in each time step. 


(5) Use the python script to merge the stored SGT data and to create the SGT database.
eg: ../examples/example_1_merge_SGT_database.py

