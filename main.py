# Importing necessary modules
try:
    from dask.distributed import Client, progress
    import dask.array as da
    import dask.dataframe as dd
    import numpy as np
    from engine import *

except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()


# Script parameters
RUN_MODE = 'local'
RANDOM_SEED = 42
INPUT_DATA = 'testing_data.csv'
LABEL_COLUMN = 'label'


# Initialize local client if requested
if RUN_MODE == 'local':
    import webbrowser
    client = Client(processes=False, threads_per_worker=1,
                n_workers=4, memory_limit='2GB')
    #webbrowser.open(client.dashboard_link)


# Initialize random number generator from Dask and numpy seed
rng = da.random.default_rng(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Read input data
data = dd.read_csv(INPUT_DATA)


# Dummy input to keep the dashboard active
a = ''#input('Press ENTER to proceed or type "Exit" to quit program:')
while a not in ['', 'Exit']:
    a = input('Press ENTER to proceed or type "Exit" to quit program:')

if a == "Exit":
    # Stop local client
    if RUN_MODE == 'local':
        client.close()
    exit()


# Run the K-means algorithm
C = get_first_sample(INPUT_DATA)
if LABEL_COLUMN != None:
    C.drop(columns=[LABEL_COLUMN], inplace=True)
C = [np.array(C)]




# Keep dashboard active until user closes it
a = ''#input('Process finished, press ENTER to exit the program:')

# Stop local client
if RUN_MODE == 'local':
    client.close()

exit()