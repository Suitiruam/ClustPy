"""
@authors:
Mauritius Klein
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustpy.utils import dip_test, dip_pval, dip_boot_samples
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

def _dipmixed(X: np.ndarray, numerical_dims:[int], categorical_dims: [int], n_clusters_init: int, random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
    """
    Start the actual DipMixed clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    numerical_dims: list of int's
        List containing the positions of numerical features within the data set X
    categorical_dims: list of int's
        List containing the positions of categorical features within the data set X
    n_clusters_init : int
        The initial number of clusters. Can also be of type np.ndarray if initial cluster centers are specified

    Returns
    -------
    tuple : (int, np.ndarray, np.ndarray)
        The final number of clusters,
        The labels as identified by DipMeans,
        The cluster centers as identified by DipMeans
    """
    X_num= X[:, numerical_dims]
    X_cat= X[:, categorical_dims]

    X_num_cspace, X_num_nspace= checkUnimodality(X_num)
    k= determine_number_of_cluster(X_num_cspace) #Maybe based both on numerical and categorical features

    X_final= map_categorial_to_numerical(X_num_cspace,X_cat, k)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_final)

    return k, kmeans.labels_, kmeans.cluster_centers_

def checkUnimodality(X_num):
    return X_num_cspace, X_num_nspace

def determine_number_of_cluster(X_num_cspace):
    return k

def map_categorical_to_numerical(X_num_cspace,X_cat,k):
    for cat_dim in :
        num_options=[]
        for num_dim in :
            uni_modal=False
            for cat_attr in :
                projected_X=
                dip_value, p_value= checkUnimodality(projected_X)
                if is unimodal:
                    unimodal=True
                else:
                    unimodal=False
                    break
            if unimodal:
                num_options.add(num_options)
    for cat_dim in :
        best_match= [None, 100]
        for num_option in num_options:
            X_option=
            costs=0
            for cat1, cat2 in :
                dip_value, p_value=checkMultimodality(X_option, X_cat1, X_cat2)
                if is unimodal:
                    mergeAttributes()
                mapping_costs=
                costs+= mapping_costs
            if costs/number_of_attribute_combinations < best_match[0][1]:
                best_match=[num_option,costs]
            else:
                continue


    return X_final


class DipMixed():
    """
    Execute the DipMeans clustering procedure.
    In contrast to other algorithms (e.g. KMeans) DipMeans is able to identify the number of clusters by itself.
    Therefore it uses the dip-dist criterion.
    It calculates the dip-value of the distances of each point within a cluster to all other points in that cluster and checks how many points are assigned a dip-value below the threshold.
    If that amount of so called split viewers is above the split_viewers_threshold, the cluster will be split using 2-Means.
    The algorithm terminates if all clusters show a unimdoal behaviour.

    Parameters
    ----------
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters_ : int
        The final number of clusters
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers

    References
    ----------
    """

    def __init__(self, random_state: np.random.RandomState = None):
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DipMixed':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DipMixed
            this instance of the DipMixed algorithm
        """
        n_clusters, labels, centers = _dipmixed(X, numerical_dims, categorical_dims, self.n_clusters_init)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self
