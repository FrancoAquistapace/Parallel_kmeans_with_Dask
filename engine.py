'''
In this file we will define all of the necessary functionality.
'''

# Importing necessary modules
try:
    from dask.distributed import Client, progress
    import dask.array as da
    import dask.dataframe as dd

except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()



# ------ K-MEANS FUNCTIONS ----------
