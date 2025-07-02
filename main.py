# Importing necessary modules
try:
    # Dask
    from dask.distributed import Client, SSHCluster, progress
    import dask.array as da
    import dask.dataframe as dd
    # Others
    import numpy as np
    import pandas as pd
    import time
    from sklearn.datasets import fetch_rcv1
    from engine import *

except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()


# Script parameters
RUN_MODE = 'ssh'
RANDOM_SEED = 42
INPUT_DATA = 'testing_data.csv'
LABEL_COLUMN = 'label'
NPARTITIONS = 1
K = 3 # Number of clusters
L = 3 # Oversampling factor
VERBOSE = 2 # Amount of information to print out (0: Nothing, 1: Only timings, 2: All information)

# Initialize local client if requested
if RUN_MODE == 'local':
    client = Client(processes=False, threads_per_worker=1,
                n_workers=4, memory_limit='2GB')

# Initialize 
elif RUN_MODE == 'ssh':
    cluster = SSHCluster(
        ["10.67.22.240", "10.67.22.240", "10.67.22.17", "10.67.22.100", "10.67.22.126"],
        connect_options={"known_hosts": None},
        worker_options={"nthreads": 2},
        scheduler_options={"port": 0, "dashboard_address": ":8787"}
    )
    client = Client(cluster)

else:
    print(f'Run mode {RUN_MODE} not supported')
    exit()


# Run main script:
# Run k-means
test_res = k_means_parallel(INPUT_DATA, K, L, client, random_seed=RANDOM_SEED, label_column=LABEL_COLUMN,
                     datatype='dataframe', npartitions=NPARTITIONS, verbose=VERBOSE)

# Close client
client.close()

exit()