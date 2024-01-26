### MPI
Contains the mpi implementation of the algorithm (without broadcasting or sharing fingerprints yet). Requires mpi4py,h5py and previous libraries from the serial version.



#### Example of how to run: 

##### mpiexec -n 2 python run_clustering_mpi.py --n 'zebra' --t1 0.3 --t2 0.6

flags:

-n    number of ranks (processes)

--n   network

--t1  initial threshold

--t2  merging threhsold 
