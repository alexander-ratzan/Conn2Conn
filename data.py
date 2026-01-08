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

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io
from concurrent.futures import ThreadPoolExecutor

def square2tri(C, tri_indices=None, k=1, return_indices=False):
    """
    Convert a single square matrix to a triangular (vectorized) matrix.
    If tri_indices is None, compute them for a C.shape[0] square matrix (above diagonal, k=1 by default).
    C should be a np.ndarray.
    """
    if tri_indices is None:
        tri_indices = np.triu_indices(C.shape[0], k=k)
    else:
        if not isinstance(tri_indices, tuple):
            # convert to tuple, since it might have been converted to a 2xEdges numpy array
            tri_indices = (tri_indices[0], tri_indices[1])
    if return_indices:
        return C[tri_indices], tri_indices
    else:
        return C[tri_indices]

def triu_indices_torch(n,k=0):
    """pytorch triu_indices doesn't work the same way so use custom function that will"""
    ia,ib=torch.triu_indices(n,n,offset=k)
    return ia,ib

def tri2square(Ctri, tri_indices=None, numroi=None, k=1, diagval=0):
    """
    Convert a 1d vectorized matrix to a square symmetrical matrix
    
    Example applying to a Nsubj x edges:
    C_list=[tri2square(Ctri[i,:],tri_indices=triu) for i in range(Ctri.shape[0])]
    or
    C_3D=np.stack([tri2square(Ctri[i,:],tri_indices=triu) for i in range(Ctri.shape[0])])
    """
    if tri_indices is None and numroi is None:
        raise Exception("Must provide either tri_indices or numroi")
    
    if tri_indices is None:
        if torch.is_tensor(Ctri):
            tri_indices=triu_indices_torch(numroi,k=k)
        else:
            tri_indices=np.triu_indices(numroi,k=k)
    else:
        if not type(tri_indices) is tuple:
            #convert to tuple, since it might have been converted to a 2xEdges numpy array
            tri_indices=(tri_indices[0],tri_indices[1])
        numroi=np.array(max(max(tri_indices[0]),max(tri_indices[1])))+1
    if torch.is_tensor(Ctri):
        C=torch.zeros(numroi,numroi,dtype=Ctri.dtype,device=Ctri.device)+torch.tensor(diagval,dtype=Ctri.dtype,device=Ctri.device)
    else:
        C=np.zeros((numroi,numroi),dtype=Ctri.dtype)+diagval
    
    C[tri_indices]=Ctri
    C[tri_indices[1],tri_indices[0]]=Ctri
    return C

def _load_single_fc_file(args):
    """Helper function to load a single FC file - designed for parallel execution"""
    subjects_dir, subj_folder, seg_string = args
    subject_id = subj_folder.replace("sub-", "")
    tsv_path = os.path.join(subjects_dir, subj_folder, "func", 
                           f"{subj_folder}_task-rest_space-fsLR_seg-{seg_string}_stat-pearsoncorrelation_relmat.tsv")
    
    if not os.path.exists(tsv_path):
        return None, None
    
    try:
        df = pd.read_csv(tsv_path, sep="\t", header=0, index_col=0)
        mat = df.values.astype(float)
        return subject_id, mat
    except Exception as e:
        print(f"[load_fc] Error loading file for subject {subject_id} at {tsv_path}: {e}. Skipping.")
        return None, None

def load_fc(parcellation='Glasser', HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/', n_jobs=None):
    """
    Load FC matrices with parallel file I/O for faster loading.
    
    Args:
        parcellation: Parcellation type (e.g., 'Glasser')
        HCP_dir: Base directory for HCP data
        n_jobs: Number of parallel workers (None = auto-detect, use all available CPUs)
    """
    subjects_dir = os.path.join(HCP_dir, "HCP1200_fMRI/xcpd-0-9-1/")
    subject_folders = [d for d in os.listdir(subjects_dir) if d.startswith("sub-")]
    subject_folders = sorted(subject_folders)
    seg_string = parcellation
    
    # Prepare arguments for parallel processing
    args_list = [(subjects_dir, subj_folder, seg_string) for subj_folder in subject_folders]
    
    # Determine number of workers
    n_jobs = min(len(subject_folders), os.cpu_count() or 4) if n_jobs is None else n_jobs
    
    fc_matrices, subject_ids = [], []
    
    # Parallel file loading (ThreadPoolExecutor is better for I/O-bound tasks)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(_load_single_fc_file, args_list)
    
    # Collect results
    for subject_id, mat in results:
        if subject_id is not None:
            subject_ids.append(int(subject_id))
            fc_matrices.append(mat)

    if len(fc_matrices) == 0:
        raise RuntimeError("No subject files found or loaded. Check HCP_dir and parcellation parameters.")
    
    # Stack matrices first
    fc_matrices = np.stack(fc_matrices, axis=0)
    
    # Pre-compute tri_indices once (all matrices have same shape) and vectorize triangle extraction
    tri_indices = np.triu_indices(fc_matrices.shape[1], k=1)
    fc_triangles = fc_matrices[:, tri_indices[0], tri_indices[1]]
    
    return subject_ids, fc_matrices, fc_triangles

def _load_single_sc_file(args):
    """Helper function to load a single SC file - designed for parallel execution"""
    HCP_dir, subj_folder, parcellation, metric_type, apply_log1p = args
    subject_id = subj_folder.replace("sub-", "")
    
    # Path to .mat file
    struct_base = f"HCP1200_DTI/qsirecon/sub-{subject_id}/anat/"
    mat_path = os.path.join(HCP_dir, struct_base, f"sub-{subject_id}_space-T1w_connectivity.mat")
    
    if not os.path.exists(mat_path):
        return None, None
    
    try:
        # Load .mat file
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)

        field_name = f"atlas_{parcellation}_{metric_type}"
        
        if field_name not in mat:
            return None, None
        
        # Extract matrix and squeeze to remove extra dimensions
        arr = np.array(mat[field_name])
        arr = np.squeeze(arr).astype(float)
        
        # Apply log1p transformation if requested
        if apply_log1p:
            arr = np.log1p(np.maximum(arr, 0))
        
        return subject_id, arr
    except Exception as e:
        print(f"[load_sc] Error loading file for subject {subject_id} at {mat_path}: {e}. Skipping.")
        return None, None

def load_sc(parcellation='Glasser', metric_type='sift_invnodevol_radius2_count_connectivity', 
            HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/', 
            apply_log1p=False, n_jobs=None):
    """
    Load SC matrices with parallel file I/O for faster loading.
    
    Args:
        parcellation: Parcellation type (e.g., 'Glasser', '4S456Parcels', 'S456')
        metric_type: SC metric type (e.g., 'radius2_count_connectivity', 
                     'radius2_meanlength_connectivity', 'sift_invnodevol_radius2_count_connectivity',
                     'sift_radius2_count_connectivity')
        HCP_dir: Base directory for HCP data
        apply_log1p: If True, apply log1p transformation to the SC matrices
        n_jobs: Number of parallel workers (None = auto-detect, use all available CPUs)
    
    Returns:
        subject_ids: List of subject IDs
        sc_matrices: Stacked SC matrices (n_subjects, n_roi, n_roi)
        sc_triangles: Upper triangle vectors (n_subjects, n_edges)
    """
    # Get subject folders from FC directory (same subjects for both FC and SC)
    subjects_dir = os.path.join(HCP_dir, "HCP1200_fMRI/xcpd-0-9-1/")
    subject_folders = [d for d in os.listdir(subjects_dir) if d.startswith("sub-")]
    subject_folders = sorted(subject_folders)
    
    # Prepare arguments for parallel processing
    args_list = [(HCP_dir, subj_folder, parcellation, metric_type, apply_log1p) for subj_folder in subject_folders]
    
    # Determine number of workers
    n_jobs = min(len(subject_folders), os.cpu_count() or 4) if n_jobs is None else n_jobs
    
    sc_matrices, subject_ids = [], []
    
    # Parallel file loading (ThreadPoolExecutor is better for I/O-bound tasks)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(_load_single_sc_file, args_list)
    
    # Collect results
    for subject_id, mat in results:
        if subject_id is not None:
            subject_ids.append(int(subject_id))
            sc_matrices.append(mat)

    if len(sc_matrices) == 0:
        raise RuntimeError("No subject files found or loaded. Check HCP_dir, parcellation, and metric_type parameters.")
    
    # Stack matrices first
    sc_matrices = np.stack(sc_matrices, axis=0)
    
    # Pre-compute tri_indices once (all matrices have same shape) and vectorize triangle extraction
    tri_indices = np.triu_indices(sc_matrices.shape[1], k=1)
    sc_triangles = sc_matrices[:, tri_indices[0], tri_indices[1]]
    
    return subject_ids, sc_matrices, sc_triangles

def load_metadata():
    """
    Loads and returns all subject IDs (as 'sub-XXXXX'), and the lists of subjects that belong to train, val, and test splits (also as 'sub-XXXXX').

    Returns:
        all_subject_ids: List of all subject IDs as strings (e.g., ['sub-100206', ...])
        train_subject_ids: List of subject IDs in the training set
        val_subject_ids: List of subject IDs in the validation set
        test_subject_ids: List of subject IDs in the test set
    """
    participants_path = "/scratch/asr655/neuroinformatics/krakencoder/example_data/HCP-YA_dataset/participants.tsv"
    participants_df = pd.read_csv(participants_path, sep='\t')
    
    all_subject_ids = participants_df['participant_id'].tolist()
    train_subjects = participants_df[participants_df["train_val_test"] == "train"]["subject"].tolist()
    val_subjects = participants_df[participants_df["train_val_test"] == "val"]["subject"].tolist()
    test_subjects = participants_df[participants_df["train_val_test"] == "test"]["subject"].tolist()

    return all_subject_ids, train_subjects, val_subjects, test_subjects

def population_average(connectivity_tensor, apply_fisher_z=False):
    """
    Compute the population average connectivity across subjects.
    
    Args:
        connectivity_tensor: torch.Tensor of shape (n_subjects, n_edges) containing connectivity values
        apply_fisher_z: If True, apply Fisher z-transform before averaging. Fisher z-transform is 
                        z = 0.5 * ln((1 + r) / (1 - r)) = atanh(r), where r is the correlation value.
                        Values are clipped to [-1, 1] before transformation.
    
    Returns:
        avg_connectivity: torch.Tensor of shape (n_edges,) containing the population average
    """
    if apply_fisher_z:
        connectivity_tensor = torch.clamp(connectivity_tensor, -1.0, 1.0) # Clip values to [-1, 1] range for Fisher z-transform (valid for correlations)
        connectivity_tensor = torch.atanh(connectivity_tensor) # Apply Fisher z-transform: z = atanh(r) = 0.5 * ln((1+r)/(1-r))
    
    # Compute mean across subjects (axis 0)
    avg_connectivity = torch.mean(connectivity_tensor, dim=0)    
    return avg_connectivity

class HCP_Dataset(Dataset):
    def __init__(self, parcellation='Glasser', source='SC', target='FC', HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
                 sc_metric_type='sift_invnodevol_radius2_count_connectivity', sc_apply_log1p=True):
        self.parcellation = parcellation
        self.all_subject_ids, self.train_subject_ids, self.val_subject_ids, self.test_subject_ids = load_metadata()
        self.fc_subject_ids, self.fc_matrices, self.fc_upper_triangles = load_fc(parcellation, HCP_dir)
        self.sc_subject_ids, self.sc_matrices, self.sc_upper_triangles = load_sc(parcellation, sc_metric_type, HCP_dir, sc_apply_log1p)

        @staticmethod
        def subject_indices_from_id(subject_list, target_subjects):
            return [i for i, subj in enumerate(subject_list) if subj in target_subjects]

        # Indices for train/val/test in SC and FC subject lists
        sc_train_indices = subject_indices_from_id(self.sc_subject_ids, self.train_subject_ids)
        sc_val_indices = subject_indices_from_id(self.sc_subject_ids, self.val_subject_ids)
        sc_test_indices = subject_indices_from_id(self.sc_subject_ids, self.test_subject_ids)

        fc_train_indices = subject_indices_from_id(self.fc_subject_ids, self.train_subject_ids)
        fc_val_indices = subject_indices_from_id(self.fc_subject_ids, self.val_subject_ids)
        fc_test_indices = subject_indices_from_id(self.fc_subject_ids, self.test_subject_ids)

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
