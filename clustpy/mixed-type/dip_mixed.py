"""
@authors:
Mauritius Klein
"""
from functools import reduce
import numpy as np
from clustpy.utils import dip_test, dip_pval, dip_boot_samples
from sklearn.utils import check_random_state
import itertools
from clustpy.partition import skinnydip

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

    #Remove numerical dimensions with unimodal feature distribution
    X_num_reduced,numerical_dims_reduced = removeUnimodals(X_num, numerical_dims)

    #Keep categorical attribute values if unimodal for all numerical dimensions or if multimodal split
    X_cat_unimodal= keep_or_split_categorical_dim_attributes(X_num_reduced, X_cat, numerical_dims_reduced, categorical_dims)


    X_cat_final=keep_or_merge_attribute_combinations(X_num_reduced, X_cat_unimodal,numerical_dims_reduced,categorical_dims)



    return k, labels_

def removeUnimodals(X_num, numerical_dims):
    numerical_dims_reduced=[]
    for num_feature in numerical_dims:
        dip_value=dip_test(X_num[:num_feature])
        if dip_value > dip_theshold:
            continue
        else:
            numerical_dims_reduced.add[num_feature]
    X_num_recuced= X_num[:,numerical_dims_reduced]
    return X_num_recuced, numerical_dims_reduced

def keep_or_split_categorical_dim_attributes(X_num_reduced, X_cat, numerical_dims_reduced, categorical_dims):
    for cat_feature in categorical_dims:
        for cat_attr in np.unique(X_cat[:, cat_feature]):
            for num_feature in numerical_dims_reduced:
                X_cat_rows = np.where(X_cat[:cat_feature] == (cat_attr))
                X_reduced = X_num_reduced[X_cat_rows, :][:, num_feature]
                dip_value=dip_test(X_reduced)
                if dip_value < dip_threshold:
                    continue
                else:
                    skinny=skinnydip()
                    skinny.fit(X_reduced,dip_threshold,outliers=False)
                    if(len(np.unique(skinny.labels)>1)):
                        #TODO SPLIT
                        print()
                    else:
                        continue
    return X_cat_reviewed

#attribute_combinations=[[[feat, attr, cluster_id],[feat, attr,cluster_id]],[[feat, attr,cluster_id],[feat, attr,cluster_id]]]
def keep_or_merge_attribute_combinations(X_num_reduced, X_cat_unimodal,numerical_dims_reduced,categorical_dims):
    #Generate inital cluster
    all_candidates=[]
    for cat_dim in categorical_dims:
        dim_candidates=[]
        for cat_attr in np.unique(X_cat_unimodal[:, cat_dim]):
            dim_candidates.apend([cat_dim,cat_attr])
        all_candidates.append(dim_candidates)
    attribute_combinations = [list(ele) for ele in list(itertools.product(*all_candidates))]
    for i in range(len(attribute_combinations)): #inital cluster ID's
        for l in range(len(attribute_combinations)):
            attribute_combinations[i][l].append(i)
    #Iterate until no cluster changes -> no more cluster are merged
    while(cluster_update==True):
        cluster_update=False
        attribute_combinations_new=[]
        #Iterate every cluster
        for combination in attribute_combinations:
            combination_indices=[]
            #Iterate attributes of cluster and use sample indices to get intersection of samples
            for i in range(len(combination)):
                cat_dim = combination[i][0]
                attr = combination[i][1]
                attr_samples= np.where(X_cat_unimodal[:, cat_dim] == attr)
                combination_indices.append(attr_samples.tolist())
            intersection_indices= list(reduce(set.intersection, [set(item) for item in combination_indices]))
            merge = True
            #Check if merge candidate is unimodal for all numerical features
            for num_feature in numerical_dims_reduced:
                X_num_cluster=X_num_reduced[intersection_indices, :][:,num_feature]
                dip_value = dip_test(X_num_cluster)
                if dip_value < dip_threshold:
                    continue
                else:
                    merge=False
                    break
            #Merge Cluster and set equal cluster ID
            if merge == True:
                new_cluster = [sublist[:2] + [combination[0][2]] for sublist in combination]
                attribute_combinations_new.append(new_cluster)
                cluster_update = True
            #Get Old cluster by ID and leave them as cluster without merging
            else:
                cluster_ids = list(set([sublist[2] for sublist in combination]))
                old_cluster1 = [sublist for sublist in combination if sublist[2] == cluster_ids[0]]
                old_cluster2 = [sublist for sublist in combination if sublist[2] == cluster_ids[1]]
                attribute_combinations_new.append(old_cluster1)
                attribute_combinations_new.append(old_cluster2)
        # Generate new combinations based on merged and old cluster
        attribute_combinations = []
        for i in range(len(attribute_combinations_new)):
            for j in range(len(attribute_combinations_new)):
                if i != j and i > j:
                    attribute_combinations.append(attribute_combinations_new[i] + attribute_combinations_new[j])
                else:
                    continue

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
