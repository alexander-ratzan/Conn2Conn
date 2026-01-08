from dataset_utils import *

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

class HCP_Dataset(Dataset):
    def __init__(self, parcellation='Glasser', source='SC', target='FC', HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
                 sc_metric_type='sift_invnodevol_radius2_count_connectivity', sc_apply_log1p=True, shuffle_seed=0):
        # choose parcellation 
        self.parcellation = parcellation
        
        # load basic covariates and train/val/test split
        self.metadata_df = load_metadata(shuffle_seed=shuffle_seed)
    
        self.all_subject_ids = self.metadata_df['subject'].tolist()
        self.train_subject_ids = self.metadata_df[self.metadata_df["train_val_test"] == "train"]["subject"].tolist()
        self.val_subject_ids = self.metadata_df[self.metadata_df["train_val_test"] == "val"]["subject"].tolist()
        self.test_subject_ids = self.metadata_df[self.metadata_df["train_val_test"] == "test"]["subject"].tolist()
        
        # load freesurfer data
        self.freesurfer_df = load_freesurfer_data()
        self.freesurfer_df = self.freesurfer_df[self.freesurfer_df['subject'].isin(self.all_subject_ids)]       
        
        # load fc and sc matrices
        self.fc_subject_ids, self.fc_matrices, self.fc_upper_triangles = load_fc(parcellation, HCP_dir)
        self.sc_subject_ids, self.sc_matrices, self.sc_upper_triangles = load_sc(parcellation, sc_metric_type, HCP_dir, sc_apply_log1p)

        # assign fc
        sc_train_indices = self.subject_indices_from_id(self.sc_subject_ids, self.train_subject_ids)
        sc_val_indices = self.subject_indices_from_id(self.sc_subject_ids, self.val_subject_ids)
        sc_test_indices = self.subject_indices_from_id(self.sc_subject_ids, self.test_subject_ids)

        fc_train_indices = self.subject_indices_from_id(self.fc_subject_ids, self.train_subject_ids)
        fc_val_indices = self.subject_indices_from_id(self.fc_subject_ids, self.val_subject_ids)
        fc_test_indices = self.subject_indices_from_id(self.fc_subject_ids, self.test_subject_ids)
    
        self.SC_train = torch.tensor(self.sc_upper_triangles[sc_train_indices], dtype=torch.float32)
        self.SC_train_avg = population_average(self.SC_train)
        self.SC_val = torch.tensor(self.sc_upper_triangles[sc_val_indices], dtype=torch.float32)
        self.SC_val_avg = population_average(self.SC_val)
        self.SC_test = torch.tensor(self.sc_upper_triangles[sc_test_indices], dtype=torch.float32)
        self.SC_test_avg = population_average(self.SC_test)

        self.FC_train = torch.tensor(self.fc_upper_triangles[fc_train_indices], dtype=torch.float32)
        self.FC_train_avg = population_average(self.FC_train, apply_fisher_z=True)
        self.FC_val = torch.tensor(self.fc_upper_triangles[fc_val_indices], dtype=torch.float32)
        self.FC_val_avg = population_average(self.FC_val, apply_fisher_z=True)
        self.FC_test = torch.tensor(self.fc_upper_triangles[fc_test_indices], dtype=torch.float32)
        self.FC_test_avg = population_average(self.FC_test, apply_fisher_z=True)
    
    @staticmethod
    def subject_indices_from_id(subject_list, target_subjects):
        return [i for i, subj in enumerate(subject_list) if subj in target_subjects]
