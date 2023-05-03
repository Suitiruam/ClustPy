"""
@authors:
Mauritius Klein
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustpy.utils import dip_test, dip_pval, dip_boot_samples
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import itertools as it

def _dipmixed(X: np.ndarray, numerical_dims:[int], categorical_dims: [int], k: int, p_value_threshold: float , random_state: np.random.RandomState) -> (int, np.ndarray, np.ndarray):
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
    X_num= X[:,numerical_dims]
    X_cat= X[:,categorical_dims]

    X_num_cspace, X_num_nspace, weights= checkUnimodality(X_num,numerical_dims,p_value_threshold)
    if k== None:
        k= determine_number_of_cluster(X_num_cspace) #Maybe based both on numerical and categorical features

    X_final, feature_weights= map_categorial_to_numerical(X_num_cspace,X_cat, k, weights,numerical_dims,categorical_dims)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_final)

    return k, kmeans.labels_, kmeans.cluster_centers_

def checkUnimodality(X_num, numerical_dims, p_value_threshold):
    clustered_features = []
    noise_features = []
    for num_dims in numerical_dims:
        dip_value=dip_test(X_num[:num_dims])
        pval = dip_pval(dip_value, np.shape(X_num)[0])
        if pval < p_value_threshold:
            clustered_features.add(num_dims)
        else:
            noise_features.add(num_dims)
    X_num_cspace=X_num[:,clustered_features]
    X_num_nspace=X_num[:,noise_features]
    return X_num_cspace, X_num_nspace



def determine_number_of_cluster(X_num_cspace):
    return k

def map_categorial_to_numerical(X_num_cspace,X_cat,k, feature_weights, numerical_dims,categorical_dims):
    for cat_dim in categorical_dims:
        num_options=[]
        for num_dim in numerical_dims:
            uni_modal=False
            for cat_attr in np.unique(X_cat[:,cat_dim]):
                X_cat_rows = np.where(X_cat[:cat_dim] == (cat_attr))
                X_reduced = X_num_cspace[X_cat_rows, :][:, num_dim]
                dip_value, p_value= checkUnimodality(X_reduced)
                if is unimodal:
                    unimodal=True
                else:
                    unimodal=False
                    break
            if unimodal:
                num_options.add(num_options)

    cat_mapping=[]
    projections=[]

    for cat_dim in categorical_dims:
        attribute_combinations= pd.Series(list(it.combinations(np.unique(X_cat[:cat_dim]),2)))
        best_projection= None
        best_projection_costs=100
        projection_weight=0
        for num_option in num_options:
            costs=0
            for combination in attribute_combinations:
                cat1, cat2= combination[0],combination[1]
                X_cat_rows=np.where(X_cat[:cat_dim] == (cat1 or cat2))
                X_option=X_num_cspace[X_cat_rows,:][:,num_option]
                dip_value, p_value=checkMultimodality(X_option)
                if is unimodal:
                    mergeAttributes()
                    mergeIterator+=1
                mapping_costs=
                costs+= mapping_costs
            feature_costs= costs/len(attribute_combinations)
            if feature_costs < best_projection_costs:
                best_projection = num_option
                best_projection_costs=feature_costs
                projection_weight=diff(best_projection_cost,feature_weights[num_option])
            else:
                continue
        if len(attributeCombinations)>1:
            cat_mapping.add(cat_dim)
            projections.add(best_projection)
            feature_weights.add(projection_weight*feature_weights[num_option])
        #else:
         #   cat_noise.add[cat_dim]

        X_cat_cspace= X_num_cspace[:projections]
        X_final=X_num_cspace+X_cat_cspace

        #Idea: calculate k based on both numerical and categorical information -> check if equal, if not use both and choose better one

    return X_final, cat_mapping, projections, feature_weights


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

    def __init__(self, random_state: np.random.RandomState = None, p_value_threshold:float=0.01):
        self.random_state = check_random_state(random_state)
        self.p_value_threshold=p_value_threshold

    def fit(self, X: np.ndarray, numerical_dims:[int], categorical_dims:[int], y: np.ndarray = None) -> 'DipMixed':
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
        n_clusters, labels, centers = _dipmixed(X, numerical_dims, categorical_dims, self.n_clusters_init, self.p_value_threshold)
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self
