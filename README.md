# stream-graph

### SERIAL 
Contains the serial implementation of the algorithm and code for experiments related to the main algorithm. Required libraries are numpy, networkx, pandas, scipy, pickle, plotly, matplotlib, sklearn. The evaluation experiments require cdlb https://cdlib.readthedocs.io/en/latest/index.html



#### To create a virtual environment with virtualenv:

virtualenv -p python3.7 python37_env

source python37_env/bin/activate

pip install cdlib, numpy, pandas 

You can also create a virtual environment with anaconda 




#### Example of how to run: 

python run_clustering_serial.py --n 'harveysept13' --t1 0.1111111111111111 --t2 0.6666666666666666

flags:
--n   network

--t1  initial threshold

--t2  merging threhsold 



### MPI
Contains the mpi implementation of the algorithm (without broadcasting or sharing fingerprints yet). Requires mpi4py,h5py and previous libraries from the serial version.



#### Example of how to run: 

mpiexec -n 2 python run_clustering_mpi.py --n 'harveysept13' --t1 0.1111111111111111 --t2 0.6666666666666666

flags:

-n    number of ranks (processes)

--n   network

--t1  initial threshold

--t2  merging threhsold 


### SPARK
Contains code for graph generation in Spark

Requires pyspark
