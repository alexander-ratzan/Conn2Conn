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
    parcellation='Glasser', hemi='both', source='SC', target='FC', shuffle_seed=0,
    sc_metric_type='sift_invnodevol_radius2_count_connectivity', sc_apply_log1p=True, 
    num_pca_components_sc=256, num_pca_components_fc=256):
        """
        Load in data for all HCP subjects
        Define features and targets
        Assign train/val/test splits
        Compute averages and PCA transforms per partition
        """
        # Choose parcellation
        self.parcellation = parcellation # Glasser or 4S456Parcels
        self.hemi = hemi # select whether to subset to a specific hemisphere or both
        self.HCP_dir = HCP_dir
        self.source = source # SC or FC
        self.target = target # SC or FC
        self.sc_metric_type = sc_metric_type
        self.sc_apply_log1p = sc_apply_log1p
        self.shuffle_seed = shuffle_seed
        self.num_pca_components_sc = num_pca_components_sc
        self.num_pca_components_fc = num_pca_components_fc

        # Load basic covariates and train/val/test split
        self.metadata_df, self.covariate_one_hot_tuple = load_metadata(shuffle_seed=shuffle_seed, age_bin_size=3)
        self.all_subject_ids = self.metadata_df['subject'].tolist()
        
        # Load freesurfer data
        self.freesurfer_df = load_freesurfer_data()
        self.freesurfer_df = self.freesurfer_df[self.freesurfer_df['subject'].isin(self.all_subject_ids)]       
            
        # Load fc and sc matrices
        self.fc_subject_ids, self.fc_matrices, self.fc_upper_triangles = load_fc(parcellation, hemi, HCP_dir)
        self.sc_subject_ids, self.sc_matrices, self.sc_upper_triangles, self.sc_r2t_matrices = load_sc(parcellation, hemi, sc_metric_type, HCP_dir, sc_apply_log1p)
        
        # Include subjects that common to metadata, FC, SC, and freesurfer dataframes
        canonical_subject_ids = sorted(set(self.all_subject_ids) & set(self.fc_subject_ids) & set(self.sc_subject_ids) & set(self.freesurfer_df['subject']))
        canonical_meta_indices = self.subject_indices_from_id(self.all_subject_ids, canonical_subject_ids)
        canonical_freesurfer_indices = self.subject_indices_from_id(self.freesurfer_df['subject'].tolist(), canonical_subject_ids)
        canonical_fc_indices = self.subject_indices_from_id(self.fc_subject_ids, canonical_subject_ids)
        canonical_sc_indices = self.subject_indices_from_id(self.sc_subject_ids, canonical_subject_ids)
        
        self.metadata_df = self.metadata_df.iloc[canonical_meta_indices]
        self.covariate_one_hot_tuple = tuple(x[canonical_meta_indices] for x in self.covariate_one_hot_tuple)
        self.freesurfer_df = self.freesurfer_df.iloc[canonical_freesurfer_indices]
        self.fc_matrices = self.fc_matrices[canonical_fc_indices]
        self.fc_upper_triangles = self.fc_upper_triangles[canonical_fc_indices]
        self.sc_matrices = self.sc_matrices[canonical_sc_indices]
        self.sc_upper_triangles = self.sc_upper_triangles[canonical_sc_indices]
        self.sc_r2t_matrices = self.sc_r2t_matrices[canonical_sc_indices]

        self.trainvaltest_partition_indices = {
            "train": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "train"]["subject"].tolist()),
            "val": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "val"]["subject"].tolist()),
            "test": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "test"]["subject"].tolist()),
        }
        
        # Compute mean and PCA for training subjects for FC and SC
        self.sc_train_avg,  self.sc_train_loadings, self.sc_train_scores = population_mean_pca(self.sc_upper_triangles[self.trainvaltest_partition_indices["train"]])
        self.fc_train_avg, self.fc_train_loadings, self.fc_train_scores = population_mean_pca(self.fc_upper_triangles[self.trainvaltest_partition_indices["train"]])
        
    @staticmethod
    def subject_indices_from_id(subject_list, target_subjects):
        return [i for i, subj in enumerate(subject_list) if subj in target_subjects]

class HCP_Partition(Dataset):
    def __init__(self, base, partition):
        """
        Args:
            base: An instance of the full dataset class that stores all data arrays and indices.
            partition: One of ["train", "val", "test"] specifying which split for this dataset.
        """
        self.base = base
        self.partition = partition
        self.indices = base.trainvaltest_partition_indices[partition] # idx list like: [2, 4, 15, 19, 21, 24...]
        
        self.source = base.source
        self.target = base.target

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sc_upper_triangles = torch.tensor(base.sc_upper_triangles, dtype=torch.float32, device=self.device)
        self.fc_upper_triangles = torch.tensor(base.fc_upper_triangles, dtype=torch.float32, device=self.device)
        
        # self.sc_train_avg = torch.tensor(base.sc_train_avg, dtype=torch.float32, device=device)
        # self.sc_train_loadings = torch.tensor(base.sc_train_loadings, dtype=torch.float32, device=device)
        # self.sc_train_scores = torch.tensor(base.sc_train_scores, dtype=torch.float32, device=device)

        # self.fc_train_avg = torch.tensor(base.fc_train_avg, dtype=torch.float32, device=device)
        # self.fc_train_loadings = torch.tensor(base.fc_train_loadings, dtype=torch.float32, device=device)
        # self.fc_train_scores = torch.tensor(base.fc_train_scores, dtype=torch.float32, device=device)


    def __getitem__(self, idx):
        """
        By default, returns a dict with:
            - 'x': upper triangle vector of the source modality
            - 'y': upper triangle vector of the target modality
        
        Later, more keys can be added (covariates, freesurfer, tracts, etc)
        """
        global_idx = self.indices[idx] # retruns index of subject in global subject list
        if self.source == "SC":
            source_data = self.sc_upper_triangles[global_idx]
        elif self.source == "FC":
            source_data = self.fc_upper_triangles[global_idx]
            
        if self.target == "SC":
            target_data = self.sc_upper_triangles[global_idx]
        elif self.target == "FC":
            target_data = self.fc_upper_triangles[global_idx]
        
        return {
            "x": source_data,
            "y": target_data,
        }
    
    def __len__(self):
        return len(self.indices)
