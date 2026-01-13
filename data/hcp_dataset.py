from typing import Any

from data.dataset_utils import *

"""
Define HCP dataset class

Desired features:
- load HCP dataset for a given inputted parcellation type
- assign global metadata for whether or not SC or FC is the target
- each subjects SC and FC files get loaded in for a given parcellation
    - accompany the parcellation with x,y,z coordinates
    - to build out: accompany the parcellation with gene expression per parcel (can be PCA or full version)
- each subjects metadata files get loaded in from xlsx with lists for demographic data (familial relationship), behavioral data, supplementary imaging information
- .mat file partitions subjects into train, val, test
    - precompute means and variances for each partition
    - precompute PCA transformations
- implement a getter than returns all relevant info on a per subject basis
    - return the input graph and the output graph (core func)
    - return relevant covariate info
"""

class HCP_Base():
    def __init__(self,
    HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
    parcellation='Glasser', source='SC', target='FC', shuffle_seed=0,
    sc_metric_type='sift_invnodevol_radius2_count_connectivity', sc_apply_log1p=True, 
    pca_components_sc=256, pca_components_fc=256):
        """
        Load in data for all HCP subjects
        Define features and targets
        Assign train/val/test splits
        Compute averages and PCA transforms per partition
        """
        # Choose parcellation
        self.parcellation = parcellation
        self.HCP_dir = HCP_dir
        self.source = source
        self.target = target
        self.sc_metric_type = sc_metric_type
        self.sc_apply_log1p = sc_apply_log1p
        self.shuffle_seed = shuffle_seed
        self.pca_components_sc = pca_components_sc
        self.pca_components_fc = pca_components_fc

        # Load basic covariates and train/val/test split
        self.metadata_df, self.covariate_one_hot_tuple = load_metadata(shuffle_seed=shuffle_seed, age_bin_size=3)
        self.all_subject_ids = self.metadata_df['subject'].tolist()
        
        # Load freesurfer data
        self.freesurfer_df = load_freesurfer_data()
        self.freesurfer_df = self.freesurfer_df[self.freesurfer_df['subject'].isin(self.all_subject_ids)]       
            
        # Load fc and sc matrices
        self.fc_subject_ids, self.fc_matrices, self.fc_upper_triangles = load_fc(parcellation, HCP_dir)
        self.sc_subject_ids, self.sc_matrices, self.sc_upper_triangles = load_sc(parcellation, sc_metric_type, HCP_dir, sc_apply_log1p)
        
        # Include subjects that common to metadata, FC, SC, and freesurfer dataframes
        canonical_subject_ids = sorted(set(self.all_subject_ids) & set(self.fc_subject_ids) & set(self.sc_subject_ids) & set(self.freesurfer_df['subject']))
        canonical_meta_indices = self.subject_indices_from_id(self.all_subject_ids, canonical_subject_ids)
        canonical_freesurfer_indices = self.subject_indices_from_id(self.freesurfer_df['subject'].tolist(), canonical_subject_ids)
        canonical_fc_indices = self.subject_indices_from_id(self.fc_subject_ids, canonical_subject_ids)
        canonical_sc_indices = self.subject_indices_from_id(self.sc_subject_ids, canonical_subject_ids)
        
        self.metadata_df = self.metadata_df.iloc[canonical_meta_indices]
        self.freesurfer_df = self.freesurfer_df.iloc[canonical_freesurfer_indices]
        self.fc_matrices = self.fc_matrices[canonical_fc_indices]
        self.fc_upper_triangles = self.fc_upper_triangles[canonical_fc_indices]
        self.sc_matrices = self.sc_matrices[canonical_sc_indices]
        self.sc_upper_triangles = self.sc_upper_triangles[canonical_sc_indices]

        self.trainvaltest_partition_indices = {
            "train": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "train"]["subject"].tolist()),
            "val": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "val"]["subject"].tolist()),
            "test": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "test"]["subject"].tolist()),
        }
        
        # Compute mean and PCA for training subjects for FC and SC
        self.sc_train_avg, self.sc_train_scores, self.sc_train_loadings = population_mean_pca(self.sc_upper_triangles[self.trainvaltest_partition_indices["train"]])
        self.fc_train_avg, self.fc_train_scores, self.fc_train_loadings = population_mean_pca(self.fc_upper_triangles[self.trainvaltest_partition_indices["train"]])
        
    @staticmethod
    def subject_indices_from_id(subject_list, target_subjects):
        return [i for i, subj in enumerate(subject_list) if subj in target_subjects]


