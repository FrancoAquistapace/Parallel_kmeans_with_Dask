'''
In this file we will define all of the necessary functionality.
'''

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



# -------- K-MEANS FUNCTIONS ----------
def get_first_sample(data_path):
    '''
    Params:
        data_path : str
            Path to a given data file.
    Output:
        Returns the first row of the given data.
    '''
    # Read only first row using pandas
    return pd.read_csv(data_path, nrows=1)


def get_XXT_term(X):
    '''
    Params:
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the self multiplication term from the
        squared distance matrix formula.
    '''
    # Turn into array
    X_da = da.array(X)
    # Get diagonal of X*X_T
    XXT = da.einsum('ij,ij -> i', X_da, X_da)
    return XXT


def partial_squared_dist_matrix(C, X):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the partial squared distance matrix, 
        evaluated over the set of points X with respect 
        to the centroids C. The partial matrix is defined
        as:
         D'^2 = 2*XC^T + CC^T
    '''
    # Turn into arrays
    X_da = da.array(X)
    
    # Calculate XC term
    XC_term = da.einsum('nj, mj -> nm', X_da, C)

    # Calculate CC term
    CC_term = da.einsum('ij,ij -> i', C, C)
    
    return -2*XC_term + CC_term
    

def cost_function(C, X, XXT):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
        XXT : Dask array or dataframe
            Array with same number of rows as X, 
            containing the X*X^T term of the
            squared distance matrix.
    Output:
        Returns the K-means cost function, evaluated
        over the set of points X with respect to the
        centroids C.
    '''
    # First, get partial squared distances
    D2 = partial_squared_dist_matrix(C, X)

    # Minimize over C axis and sum
    D2_min_sum = da.sum(da.min(D2, axis=1))

    # Add to XXT sum and return
    return da.sum(XXT) + D2_min_sum


def sample_new_centroids(C, X, XXT, phi, l):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
        XXT : Dask array or dataframe
            Array with same number of rows as X, 
            containing the X*X^T term of the
            squared distance matrix.
        phi : float
            Current cost function for the data X
            and the centroids C.
        l : float or int
            Oversampling factor, must be greater 
            than zero.
    Output:
        Returns new centroids from X, sampled with
        probability:
             p_x = l * D^2 / phi
        where D^2 is the squared distance from x to
        C and phi is the current cost function.
    '''
    # Get D^2
    partial_D_sq = partial_squared_dist_matrix(C, X)
    D_sq = da.add(XXT, da.min(partial_D_sq, axis=1))

    # Get sampling probabilities
    p_X = (l * D_sq / phi).compute_chunk_sizes()
    
    # Draw random numbers between 0 and 1
    rand_nums = da.random.uniform(size=p_X.shape)

    # Get new centroid indexes
    C_prime_idx = rand_nums < p_X

    # Gather new centroids from the data
    C_prime = X.loc[C_prime_idx]
    return C_prime

def get_cluster_classification(C, X, XXT):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the corresponding centroid in C
        for each sample in X.
    '''
    # Get partial squared distances
    partial_D_sq = partial_squared_dist_matrix(C, X)

    # Select the correct centroid from arg min
    C_ids = da.argmin(partial_D_sq, axis=1)
    return C_ids

def get_centroid_weights(labels):
    '''
    Params:
        labels : array
            Set of labels indicating the centroid of
            each data point.
    Output:
        Returns the weight of each centroid, defined 
        as the number of samples in the data closer
        to that centroid than to any other centroid.
    '''
    unique_labels, counts = da.unique(da.array(labels), return_counts=True)
    return unique_labels, counts


def compute_centroids(X, weights, labels, dims):
    '''
    Params:
        X : array
            Array containing data points.
        weights : array
            Weight for each data point.
        labels : array
            Set of labels indicating the cluster of
            each data point.
        dims : int
            Number of dimensions of the given data.
    Output:
        Returns the centroids for the given data and
        the given partition.
    '''    
    # Turn data into Dask array
    X = da.array(X)
    weights = da.array(weights)
    
    # Get unique labels
    unique_l = da.unique(da.array(labels)).compute_chunk_sizes()

    # Init new centroids
    C = da.array(np.zeros(shape=(len(unique_l), dims)))

    # Operate for each unique cluster
    for i, idx in enumerate(unique_l):
        # Get data
        C_data = X[labels == idx, :].compute_chunk_sizes()
        data_weights = weights[labels == idx].compute_chunk_sizes()

        # Calculate weighted mean as new centroid
        new_C = da.average(C_data, axis=0, weights=data_weights)

        # Store new centroid
        C[i,:] += new_C
        
    return C


# Complete k-means pipeline
def k_means_parallel(path, k, l, client, random_seed=None, label_column=None,
                     datatype='dataframe', npartitions=1, chunk_size=1000,
                     verbose=2):
    '''
    Params:
        path : str
            Path to the file containing the input data. If "rcv1", then
            the RCV1 dataset is downloaded inside the function.
        k : int
            Number of clusters to use.
        l : int or float
            Oversampling factor for the K-means|| initialization method.
        client : dask Client
            Client from which to run all operations.
        random_seed : int (optional)
            Seed for the random number generators. Not used by default.
        label_column : str (optional)
            Name of the column containing the labels of the data. Not
            used by default.
        datatype : str (optional)
            Either "dataframe" (default) or "array". Indicates how to
            process the input data.
        npartitions : int (optional)
            Number of partitions to use if the data is processed as a
            dataframe. Default is 1.
        chunk_size : int (optional)
            Chunk size to use if the data is processed as an array.
            Default is 1000.
        verbose : int (optional)
            Amount of information to print. If 0, no information is 
            printed. If 1, only timing information is printed. If 2
            (default), all information is printed.
    Output:
        Returns a dictionary with the following elements:
            - centroids : array       -> Centroid positions
            - cluster_labels : array  -> Cluster assignments
            - timing : dict           -> Performance information
    '''
    # Begin overall timing
    t1 = time.time()
    
    # Initialize random number generator from Dask and numpy seed
    if random_seed != None:
        rng = da.random.default_rng(random_seed)
        np.random.seed(random_seed)

    # Read input data
    t_data = time.time()
    if verbose == 2:
        print('Reading input data')

    if path not in ['rcv1']:
        data = pd.read_csv(path)
        data_shape = data.shape
        
    elif path == 'rcv1':
        # Load RCV1 dataset
        rcv1 = fetch_rcv1()
        data = rcv1.data
        data_shape = data.shape
    dt_data = time.time() - t_data

    # Separate labels from input
    t_data_process = time.time()
    if label_column != None:
        # Labels
        future = client.scatter(data[label_column])  # send labels to one worker
        y_true = dd.from_delayed([future], meta=data[label_column])  # build dask.dataframe on remote data
        y_true = y_true.repartition(npartitions=npartitions).persist()  # split
        client.rebalance(y_true)  # spread around all of your workers
    
        # Input
        X_width = data_shape[1]-1
        X = data.drop(columns=[label_column])
        future = client.scatter(X) # send data to one worker
        X = dd.from_delayed([future], meta=X)  # build dask.dataframe on remote data
        X = X.repartition(npartitions=npartitions).persist()  # split
        client.rebalance(X)  # spread around all of your workers
        
    else:
        # Only input
        X_width = data_shape[1]
        X = data
        future = client.scatter(X) # send data to one worker
        X = dd.from_delayed([future], meta=X, shape=data_shape)  # build dask.dataframe on remote data
        X = X.repartition(npartitions=npartitions).persist()  # split
        client.rebalance(X)  # spread around all of your workers
        
    dt_data_process = time.time() - t_data_process
    if verbose > 0:
        print(f'Data loaded and processed in {round(dt_data_process + dt_data, 1)} s')

    # Run the K-means algorithm:
    # Get first sample as initial centroid
    t_first_centroid = time.time()
    if path not in ['rcv1']:
        first_sample = get_first_sample(path)
    elif path == 'rcv1':
        first_sample = data[0,:].toarray()
        
    if label_column != None:
        first_sample = first_sample.drop(columns=[label_column])
    C = da.array([np.array(first_sample).flatten()])

    dt_first_centroid = time.time() - t_first_centroid
    
    # Calculate constant XXT term, 
    # also persist since we are going to reuse it 
    t_xxt = time.time()
    XXT = get_XXT_term(X).persist()
    dt_xxt = time.time() - t_xxt
    
    # Get initial cost function
    t_phi_init = time.time()
    phi_init = cost_function(C, X, XXT).compute()
    dt_phi_init = time.time() - t_phi_init
    
    # Get number of iterations of the || algorithm
    O_log_phi = round(np.log(phi_init))
    
    # Init current cost
    phi = phi_init.copy()

    # Proceed with main || loop
    t_parallel_init = time.time()
    if verbose == 2:
        print('\nRunning K-means|| initialization:')
    for i in range(O_log_phi):
        if verbose == 2:
            print(f'Iteration {i+1} of {O_log_phi}')
            
        # Sample new centroids
        C_prime = sample_new_centroids(C, X, XXT, phi, L)
    
        # Add to the current centroids
        C = da.vstack([C, C_prime]).compute()
    
        # Calculate new cost and update current
        phi = cost_function(C, X, XXT).compute()
    
    if verbose == 2:
        # Print number of final centroids from ||
        print('\nNumber of initialized centroids:', C.shape[0])
        
        # Print initial vs. final cost
        print('Cost before initialization:', phi_init)
        print('Cost after initialization:', phi)

    dt_parallel_init = time.time() - t_parallel_init
    if verbose > 0:
        print(f'K-means|| initialization finished in {round(dt_parallel_init, 1)} s')
    
    # Get the weight of each centroid
    if verbose == 2:
        print('\nCalculating centroid weights')
    t_weight_calc = time.time()
    X_labels = get_cluster_classification(C, X, XXT).compute_chunk_sizes()
    used_C, w_C = get_centroid_weights(X_labels)
    used_C = used_C.compute()
    w_C = w_C.compute()

    dt_weight_calc = time.time() - t_weight_calc
    if verbose > 0:
        print(f'Centroid weight calculation finished in {round(dt_weight_calc, 1)} s')

    # Proceed with Lloyd's algorithm on the centroids
    t_lloyd = time.time()
    if verbose == 2:
        print('\nClustering centroids')
    
    # Initialize k final centroids, as the k-th heaviest
    # centroids from the previous step
    C_f = C[np.isin(w_C, np.sort(w_C, )[len(w_C)-K:])]
    
    # Calculate XXT for centroids
    CCT =  get_XXT_term(C).persist()
    
    # Perform iterative adjustments
    lloyd_done = False
    N_lloyd_steps = 0
    while not lloyd_done:
        # Save old labels (after first iteration)
        if N_lloyd_steps > 0:
            old_labels = C_labels.copy()
        
        # Calculate current clustering
        C_labels = get_cluster_classification(C_f, C, CCT).persist()
    
        # Compute new centroids from mean within clusters
        C_f = compute_centroids(C, w_C, C_labels, X_width).compute()
        
        # Check for termination condition (after first iteration)
        if N_lloyd_steps > 0:
            different_labels = da.sum(old_labels != C_labels).compute()
            if different_labels == 0:
                lloyd_done = True
    
        # Increase step counter
        N_lloyd_steps += 1
    dt_lloyd = time.time() - t_lloyd
    
    if verbose == 2:
        print(f'Centroid clustering finished after {N_lloyd_steps} iterations and {round(dt_lloyd)} s')
    elif verbose == 1:
        print(f'\nLloyd algorithm finished in {round(dt_lloyd)} s')
    
    # Compute final labels
    if verbose == 2:
        print('\nCalculating final labels')
    t_final_labels = time.time()
    final_labels = get_cluster_classification(C_f, X, XXT).compute()
    dt_final_labels = time.time() - t_final_labels
    if verbose > 0:
        print(f'Final labels calculated in {round(dt_final_labels)} s')

    # Finish timing
    dt_total = time.time() - t1
    if verbose > 0:
        print(f'\nProcess finished in {round(dt_total, 1)} s')

    # Build performance info
    timing = {'total': dt_total,
              'data_input': dt_data,
              'data_processing': dt_data_process,
              'first_centroid': dt_first_centroid,
              'xxt': dt_xxt,
              'phi_init': dt_phi_init, 
              'parallel_init': dt_parallel_init, 
              'weight_calc': dt_weight_calc, 
              'lloyd' : dt_lloyd, 
              'final_labels': dt_final_labels}
    
    # Gather output info into a dict and return
    output_info = {'centroids': C_f,
                   'cluster_labels': final_labels,
                   'timing': timing}
    return output_info