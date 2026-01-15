import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io
from scipy.stats import pearsonr
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

def tri_corr(mat1, mat2):
    # extract upper triangle without diagonal, flatten and correlate
    iu = np.triu_indices_from(mat1, k=1)
    return pearsonr(mat1[iu], mat2[iu])[0]

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