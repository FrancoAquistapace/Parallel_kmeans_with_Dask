# Importing necessary modules
try:
    from dask.distributed import Client, progress
    import dask.array as da
except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()


# Script parameters
RUN_MODE = 'local'


# Initialize local client if requested
if RUN_MODE == 'local':
    client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')

# Create random array   
x = da.random.random((10000, 10000), chunks=(1000, 1000))

y = x + x.T
z = y[::2, 5000:].mean(axis=1)

print(z.compute())

#exit()