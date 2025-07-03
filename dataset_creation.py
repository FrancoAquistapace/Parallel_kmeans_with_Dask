import numpy as np
import pandas as pd

def create_blobs(n_samples, n_features, centers, random_state=None):
    '''
    Params:
        n_samples : int
            Number of samples to create.
        n_features : int
            Dimensionality of the samples.
        centers : int
            Number of distinct Gaussian centers.
        random_state : int (optional)
            Seed for the numpy random generator.
    Output:
        Returns a numpy array (n_samples, n_features),
        with samples drawn from Gaussian distributions, 
        with different mean and standard deviation 1.
    '''
    # Init seed if given
    if random_state != None:
        np.random.seed(random_state)

    # Create centers
    u = np.random.uniform(-2, 2, size=(centers, n_features))

    # Create Gaussian samples
    samples = np.random.normal(size=(n_samples, n_features))

    # Displace to randomly selected center
    labels = np.random.randint(centers, size=n_samples)
    samples += u[labels,:]
    return samples, labels



# Data parameters
N_SAMPLES = int(1e5)
N_FEATURES = 50
CENTERS = 5
RS = 42
LABEL_COLUMN = 'label'
PATH = './test_blobs_50dim_5c_1e5.csv'

synt_data, y_true = create_blobs(N_SAMPLES, N_FEATURES, CENTERS, RS)
synt_data = pd.DataFrame(synt_data)
synt_data[LABEL_COLUMN] = y_true
synt_data.to_csv(PATH, index=False)


# Finish process
exit()