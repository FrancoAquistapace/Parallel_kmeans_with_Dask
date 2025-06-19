# Importing necessary modules
try:
    from dask.distributed import Client, progress
    import dask.array as da
    import dask.dataframe as dd
    from engine import *

except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()


# Script parameters
RUN_MODE = 'local'
INPUT_DATA = 'testing_data.csv'


# Initialize local client if requested
if RUN_MODE == 'local':
    client = Client(processes=False, threads_per_worker=1,
                n_workers=4, memory_limit='2GB')

# Read input data
data = dd.read_csv(INPUT_DATA)
print(data)

# Stop local client
if RUN_MODE == 'local':
    client.close()

exit()