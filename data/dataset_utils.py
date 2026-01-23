import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io
from data.data_utils import *
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA


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

def load_fc(parcellation='Glasser', hemi='both', HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/', n_jobs=None):
    """
    Load FC matrices with parallel file I/O for faster loading.
    
    Args:
        parcellation: Parcellation type (e.g., 'Glasser')
        hemi: Hemisphere to subset to (e.g., 'left', 'right', 'both')
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
    
    # eventually make dynamic
    roi_df = pd.read_csv(f'/scratch/asr655/neuroinformatics/Conn2Conn/data/atlas_info/{parcellation}_dseg_reformatted.csv')
    
    if hemi == 'left':
        roi_mask = roi_df['hemisphere'].str.contains('L')
    elif hemi == 'right':
        roi_mask = roi_df['hemisphere'].str.contains('R')
    elif hemi == 'both':
        roi_mask = roi_df['hemisphere'].str.contains('L') | roi_df['hemisphere'].str.contains('R')
    else:
        roi_mask = np.ones(len(roi_df), dtype=bool)  # fallback: use all

    roi_indices = np.where(roi_mask)[0]
    
    for subject_id, mat in results:
        if subject_id is not None:
            if roi_indices is not None: # Subset to desired hemisphere (square matrix of roi_indices)
                mat = mat[np.ix_(roi_indices, roi_indices)]
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
    """Helper function to load a single SC file and corresponding region2tract npy - designed for parallel execution"""
    HCP_dir, subj_folder, parcellation, metric_type, apply_log1p = args
    subject_id = subj_folder.replace("sub-", "")

    # Path to .mat file
    struct_base = f"HCP1200_DTI/qsirecon/sub-{subject_id}/anat/"
    mat_path = os.path.join(HCP_dir, struct_base, f"sub-{subject_id}_space-T1w_connectivity.mat")

    # Path to region2tract .npy file
    r2t_path = os.path.join(
        HCP_dir, "HCP1200_DTI/qsirecon", f"sub-{subject_id}", "anat", "tract-to-region",
        f"sub-{subject_id}_region2tract_{parcellation}_n=66.npy"
    )
    
    # Note: 66 tracts based on common subset across subjects in original tract-to-region csvs
    tract_list = ['AssociationArcuateFasciculusL', 'AssociationArcuateFasciculusR', 'AssociationCingulumL', 'AssociationCingulumR', 'AssociationExtremeCapsuleL', 'AssociationExtremeCapsuleR', 'AssociationFrontalAslantTractL', 'AssociationFrontalAslantTractR', 'AssociationHippocampusAlveusL', 'AssociationHippocampusAlveusR', 'AssociationInferiorFrontoOccipitalFasciculusL', 'AssociationInferiorFrontoOccipitalFasciculusR', 'AssociationInferiorLongitudinalFasciculusL', 'AssociationInferiorLongitudinalFasciculusR', 'AssociationMiddleLongitudinalFasciculusL', 'AssociationMiddleLongitudinalFasciculusR', 'AssociationParietalAslantTractL', 'AssociationParietalAslantTractR', 'AssociationSuperiorLongitudinalFasciculusL', 'AssociationSuperiorLongitudinalFasciculusR', 'AssociationUncinateFasciculusL', 'AssociationUncinateFasciculusR', 'AssociationVerticalOccipitalFasciculusL', 'AssociationVerticalOccipitalFasciculusR', 'CerebellumCerebellumL', 'CerebellumCerebellumR', 'CerebellumInferiorCerebellarPeduncleL', 'CerebellumInferiorCerebellarPeduncleR', 'CerebellumMiddleCerebellarPeduncle', 'CerebellumSuperiorCerebellarPeduncle', 'CerebellumVermis', 'CommissureCorpusCallosum', 'CranialNerveCNIIIL', 'CranialNerveCNIIIR', 'CranialNerveCNIIL', 'CranialNerveCNIIR', 'CranialNerveCNVIIIL', 'CranialNerveCNVIIIR', 'CranialNerveCNVL', 'CranialNerveCNVR', 'ProjectionBasalGangliaAcousticRadiationL', 'ProjectionBasalGangliaAcousticRadiationR', 'ProjectionBasalGangliaAnsaLenticularisL', 'ProjectionBasalGangliaAnsaLenticularisR', 'ProjectionBasalGangliaAnsaSubthalamicaL', 'ProjectionBasalGangliaAnsaSubthalamicaR', 'ProjectionBasalGangliaCorticostriatalTractL', 'ProjectionBasalGangliaCorticostriatalTractR', 'ProjectionBasalGangliaFasciculusLenticularisL', 'ProjectionBasalGangliaFasciculusLenticularisR', 'ProjectionBasalGangliaFasciculusSubthalamicusL', 'ProjectionBasalGangliaFasciculusSubthalamicusR', 'ProjectionBasalGangliaFornixL', 'ProjectionBasalGangliaFornixR', 'ProjectionBasalGangliaOpticRadiationL', 'ProjectionBasalGangliaOpticRadiationR', 'ProjectionBasalGangliaThalamicRadiationL', 'ProjectionBasalGangliaThalamicRadiationR', 'ProjectionBrainstemCorticopontineTractL', 'ProjectionBrainstemCorticopontineTractR', 'ProjectionBrainstemCorticospinalTractL', 'ProjectionBrainstemCorticospinalTractR', 'ProjectionBrainstemMedialForebrainBundleL', 'ProjectionBrainstemMedialForebrainBundleR', 'ProjectionBrainstemNonDecussatingDentatorubrothalamicTractL', 'ProjectionBrainstemNonDecussatingDentatorubrothalamicTractR']

    if not os.path.exists(mat_path):
        return None, None, None

    try:
        # Load .mat file
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        field_name = f"atlas_{parcellation}_{metric_type}"

        if field_name not in mat:
            return None, None, None

        # Extract matrix and squeeze to remove extra dimensions
        arr = np.array(mat[field_name])
        arr = np.squeeze(arr).astype(float)

        # Apply log1p transformation if requested
        if apply_log1p:
            arr = np.log1p(np.maximum(arr, 0))

        # Try to load the region2tract matrix
        if os.path.exists(r2t_path):
            r2t_mat = np.load(r2t_path)
        else:
            r2t_mat = None

        return subject_id, arr, r2t_mat
    except Exception as e:
        print(f"[load_sc] Error loading file for subject {subject_id} at {mat_path}: {e}. Skipping.")
        return None, None, None

def load_sc(parcellation='Glasser', hemi='both', metric_type='sift_invnodevol_radius2_count_connectivity', 
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
        sc_r2t_matrices: Stacked region2tract matrices (n_subjects, region, t)
    """
    # Get subject folders from FC directory (same subjects for both FC and SC)
    subjects_dir = os.path.join(HCP_dir, "HCP1200_fMRI/xcpd-0-9-1/")
    subject_folders = [d for d in os.listdir(subjects_dir) if d.startswith("sub-")]
    subject_folders = sorted(subject_folders)

    # Prepare arguments for parallel processing
    args_list = [(HCP_dir, subj_folder, parcellation, metric_type, apply_log1p) for subj_folder in subject_folders]

    # Determine number of workers
    n_jobs = min(len(subject_folders), os.cpu_count() or 4) if n_jobs is None else n_jobs

    sc_matrices, subject_ids, sc_r2t_matrices = [], [], []

    # Parallel file loading (ThreadPoolExecutor is better for I/O-bound tasks)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(_load_single_sc_file, args_list)

    # Load ROI info and construct hemisphere mask (mimic FC loading logic for subsetting)
    roi_df = pd.read_csv(f'/scratch/asr655/neuroinformatics/Conn2Conn/data/atlas_info/{parcellation}_dseg_reformatted.csv')

    if hemi == 'left':
        roi_mask = roi_df['hemisphere'].str.contains('L')
    elif hemi == 'right':
        roi_mask = roi_df['hemisphere'].str.contains('R')
    elif hemi == 'both':
        roi_mask = roi_df['hemisphere'].str.contains('L') | roi_df['hemisphere'].str.contains('R')
    else:
        roi_mask = np.ones(len(roi_df), dtype=bool)  # fallback: use all

    roi_indices = np.where(roi_mask)[0]

    # Collect results robustly, skip if data not loaded correctly
    for subject_id, mat, r2t in results:
        if subject_id is not None and mat is not None:
            # Subset to desired hemisphere if possible (square matrix of roi_indices)
            if roi_indices is not None:
                mat = mat[np.ix_(roi_indices, roi_indices)]
                r2t = r2t[roi_indices, :]
            subject_ids.append(int(subject_id))
            sc_matrices.append(mat)
            sc_r2t_matrices.append(r2t)

    if len(sc_matrices) == 0:
        raise RuntimeError("No subject files found or loaded. Check HCP_dir, parcellation, and metric_type parameters.")

    # Stack matrices
    sc_matrices = np.stack(sc_matrices, axis=0)

    # Stack region2tract matrices; only if all loaded successfully, else set None
    if all(x is not None for x in sc_r2t_matrices):
        sc_r2t_matrices = np.stack(sc_r2t_matrices, axis=0)
    else:
        sc_r2t_matrices = None

    # Pre-compute tri_indices once (all matrices have same shape) and vectorize triangle extraction
    tri_indices = np.triu_indices(sc_matrices.shape[1], k=1)
    sc_triangles = sc_matrices[:, tri_indices[0], tri_indices[1]]

    return subject_ids, sc_matrices, sc_triangles, sc_r2t_matrices

def load_freesurfer_data():
    """
    Load FreeSurfer surface data from HCP1200_UNRESTRICTED.csv.
    Extracts all columns starting with 'FS_' which contain volume and surface area
    measurements for different cortical structures.
    
    Returns:
        freesurfer_df: DataFrame containing FreeSurfer data with columns:
            - subject: Subject ID as integer
            - All columns starting with 'FS_': FreeSurfer measurements
    """
    unrestricted_path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv"
    
    # Load unrestricted CSV
    unrestricted_df = pd.read_csv(unrestricted_path)
    
    # Extract Subject column and all columns starting with 'FS_'
    fs_columns = ['Subject'] + [col for col in unrestricted_df.columns if col.startswith('FS_')]
    freesurfer_df = unrestricted_df[fs_columns].copy()
    
    # Change 'Subject' to int in place and rename to 'subject'
    freesurfer_df['Subject'] = freesurfer_df['Subject'].astype(int)
    freesurfer_df = freesurfer_df.rename(columns={'Subject': 'subject'})
    
    return freesurfer_df

def load_metadata(shuffle_seed=0, age_bin_size=2):
    """
    Loads and merges metadata from participants.tsv and HCP1200_RESTRICTED.csv.
    
    Args:
        shuffle_seed: Random seed for train/val/test split generation. 
                      If shuffle_seed=0, uses the original split from participants.tsv.
                      If shuffle_seed != 0, generates a new family-preserving split using that seed.
        age_bin_size: Bin size (years) for age windowing. If None, no bin column is added.
    
    Returns:
        metadata_df: DataFrame containing all subject metadata with columns:
            - subject: Subject ID as integer
            - train_val_test: Split assignment (train/val/test)
            - age: Age in years
            - age_{age_bin_size}y_bin: Age bin label (if age_bin_size is provided; label is left edge of bin)
            - sex: Sex (M/F)
            - Race_Ethnicity: Combined race and ethnicity information
            - Family_Relation: Family relationship category (MZ, DZ, NotTwin, NoRelation)
                - MZ: Monozygotic twins (from ZygosityGT or ZygositySR)
                - DZ: Dizygotic twins (from ZygosityGT or ZygositySR)
                - NotTwin: Siblings or other family members (share Family_ID with others)
                - NoRelation: No family relation (unique Family_ID or missing data)
            - Family_ID: Family identifier
        covariate_one_hot_tuple: Tuple of (age_bin_one_hot, sex_one_hot, race_ethnicity_one_hot) numpy arrays
    """
    participants_path = "/scratch/asr655/neuroinformatics/Conn2Conn/krakencoder/example_data/HCP-YA_dataset/participants.tsv"
    restricted_path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_RESTRICTED.csv"
    
    # Load participants.tsv
    participants_df = pd.read_csv(participants_path, sep='\t')
    
    # Load restricted CSV
    restricted_df = pd.read_csv(restricted_path)
    
    # Merge on subject ID (participants_df has 'subject' column, restricted_df has 'Subject' column)
    # Convert both to same type for merging
    participants_df['subject'] = participants_df['subject'].astype(int)
    restricted_df['Subject'] = restricted_df['Subject'].astype(int)
    
    # Merge the dataframes (include both ZygosityGT and ZygositySR)
    metadata_df = participants_df.merge(
        restricted_df[['Subject', 'ZygosityGT', 'ZygositySR', 'Family_ID', 'Race', 'Ethnicity']],
        left_on='subject',
        right_on='Subject',
        how='left'
    )
    
    # Drop the duplicate Subject column
    metadata_df = metadata_df.drop(columns=['Subject'])
    
    # Create Family_Relation column based on the specified logic
    # Treat any empty string or whitespace-only string in ZygosityGT as missing (in addition to np.nan)
    family_rel_initial = metadata_df['ZygosityGT'].astype(str).str.strip()
    family_rel_initial = family_rel_initial.replace('', np.nan)
    family_rel_initial = family_rel_initial.replace('nan', np.nan)
    metadata_df['Family_Relation'] = family_rel_initial.copy()
    
    # For missing values, fill from ZygositySR (mapping 'NotMZ' to 'DZ')
    missing_mask = metadata_df['Family_Relation'].isna()
    zygosity_sr_mapped = metadata_df.loc[missing_mask, 'ZygositySR'].astype(str).str.strip().replace({'NotMZ': 'DZ', '': np.nan, 'nan': np.nan})
    metadata_df.loc[missing_mask, 'Family_Relation'] = zygosity_sr_mapped
    
    # For subjects marked as NotTwin, check Family_ID to determine if they're siblings or NoRelation
    not_twin_mask = metadata_df['Family_Relation'] == 'NotTwin'
    if not_twin_mask.any():
        family_counts = metadata_df['Family_ID'].value_counts()
        for idx in metadata_df[not_twin_mask].index:
            family_id = metadata_df.loc[idx, 'Family_ID']
            # if shared with others, keep as NotTwin, else set as NoRelation
            if pd.notna(family_id) and family_counts.get(family_id, 0) > 1:
                metadata_df.loc[idx, 'Family_Relation'] = 'NotTwin'
            else:
                metadata_df.loc[idx, 'Family_Relation'] = 'NoRelation'
    
    # Handle any remaining missing values
    metadata_df['Family_Relation'] = metadata_df['Family_Relation'].fillna('NoRelation')
    
    # Create Race_Ethnicity column by concatenating Race and Ethnicity with an underscore
    metadata_df["Race_Ethnicity"] = (
        metadata_df["Race"].astype(str) + "_" + metadata_df["Ethnicity"].astype(str)
    )
    # Drop the original Race and Ethnicity columns
    metadata_df = metadata_df.drop(columns=["Race", "Ethnicity"])

    # If random_seed != 0, generate a new family-preserving split
    if shuffle_seed != 0:
        metadata_df = generate_train_val_test(metadata_df, random_seed=shuffle_seed)

    # Concise age-binning and column selection
    if age_bin_size is not None:
        bin_col = f"age_{age_bin_size}y_bin"
        bins = np.arange(metadata_df['age'].min(), metadata_df['age'].max() + age_bin_size, age_bin_size)
        labels = [f"{int(bins[i])}-{int(bins[i] + age_bin_size - 1)}" for i in range(len(bins)-1)]
        metadata_df.insert(
            metadata_df.columns.get_loc('age') + 1,
            bin_col,
            pd.cut(metadata_df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        )
    
    cols = ['subject', 'train_val_test', 'age']
    if age_bin_size is not None:
        cols += [bin_col]
    cols += ['sex', 'Race_Ethnicity', 'Family_Relation', 'Family_ID']
    metadata_df = metadata_df[cols]

    # --- Covariate one-hot tuple ---
    # Only construct tuple if all needed columns are present
    # See toy.ipynb for structure/intent of this code
    if age_bin_size is not None:
        age_bin_oh_np = pd.get_dummies(metadata_df[bin_col], dtype=np.float32).to_numpy()
    else:
        age_bin_oh_np = None
    sex_oh_np = pd.get_dummies(metadata_df['sex'], dtype=np.float32).to_numpy()
    race_eth_oh_np = pd.get_dummies(metadata_df['Race_Ethnicity'], dtype=np.float32).to_numpy()
    covariate_one_hot_tuple = (age_bin_oh_np, sex_oh_np, race_eth_oh_np)

    return metadata_df, covariate_one_hot_tuple

def generate_train_val_test(metadata_df, random_seed=42):
    """
    Generate new train/val/test splits ensuring that all subjects with the same Family_ID
    are assigned to the same partition. Maintains similar proportions to the original split.
    
    Args:
        metadata_df: DataFrame with 'train_val_test' and 'Family_ID' columns
        random_seed: Random seed for deterministic splitting
    
    Returns:
        metadata_df: DataFrame with updated 'train_val_test' column
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get original split proportions
    original_counts = metadata_df['train_val_test'].value_counts()
    total_subjects = len(metadata_df)
    train_prop = original_counts.get('train', 0) / total_subjects
    val_prop = original_counts.get('val', 0) / total_subjects
    test_prop = original_counts.get('test', 0) / total_subjects
    
    # Group subjects by Family_ID
    # For subjects with missing Family_ID, treat each as a unique "family"
    metadata_df_copy = metadata_df.copy()
    metadata_df_copy['Family_ID_filled'] = metadata_df_copy['Family_ID'].fillna(
        metadata_df_copy['subject'].apply(lambda x: f'SINGLE_{x}')
    )
    
    # Get unique families and their sizes
    family_groups = metadata_df_copy.groupby('Family_ID_filled')
    unique_families = list(family_groups.groups.keys())
    family_sizes = {fam: len(group) for fam, group in family_groups}
    
    # Shuffle families deterministically
    np.random.shuffle(unique_families)
    
    # Calculate target number of subjects for each partition based on proportions
    total_subjects_target = sum(family_sizes.values())
    
    target_train_subjects = int(np.round(train_prop * total_subjects_target))
    target_val_subjects = int(np.round(val_prop * total_subjects_target))
    target_test_subjects = total_subjects_target - target_train_subjects - target_val_subjects
    
    # Start with all families assigned to training
    train_families = unique_families.copy()
    val_families = []
    test_families = []
    
    train_subjects_count = total_subjects_target
    val_subjects_count = 0
    test_subjects_count = 0
    
    # Iteratively assign families from train to val/test to match target proportions
    for family_id in unique_families:
        family_size = family_sizes[family_id]
        
        # Calculate total deviation from target for each option
        # Option 1: Keep in train
        keep_train_deviation = abs(train_subjects_count - target_train_subjects) + \
                              abs(val_subjects_count - target_val_subjects) + \
                              abs(test_subjects_count - target_test_subjects)
        
        # Option 2: Move to val
        move_val_deviation = abs((train_subjects_count - family_size) - target_train_subjects) + \
                            abs((val_subjects_count + family_size) - target_val_subjects) + \
                            abs(test_subjects_count - target_test_subjects)
        
        # Option 3: Move to test
        move_test_deviation = abs((train_subjects_count - family_size) - target_train_subjects) + \
                             abs(val_subjects_count - target_val_subjects) + \
                             abs((test_subjects_count + family_size) - target_test_subjects)
        
        # Choose the option that minimizes total deviation
        if move_val_deviation <= move_test_deviation and move_val_deviation < keep_train_deviation:
            # Move to val
            train_families.remove(family_id)
            val_families.append(family_id)
            train_subjects_count -= family_size
            val_subjects_count += family_size
        elif move_test_deviation < keep_train_deviation:
            # Move to test
            train_families.remove(family_id)
            test_families.append(family_id)
            train_subjects_count -= family_size
            test_subjects_count += family_size
        # Otherwise, keep in train (no action needed)
    
    # Create mapping from family to partition
    family_to_partition = {}
    for fam in train_families:
        family_to_partition[fam] = 'train'
    for fam in val_families:
        family_to_partition[fam] = 'val'
    for fam in test_families:
        family_to_partition[fam] = 'test'
    
    # Assign all subjects in each family to the same partition
    metadata_df_copy['train_val_test'] = metadata_df_copy['Family_ID_filled'].map(family_to_partition)
    
    # Drop the temporary Family_ID_filled column
    metadata_df_copy = metadata_df_copy.drop(columns=['Family_ID_filled'])
    
    # Update the original metadata_df
    metadata_df['train_val_test'] = metadata_df_copy['train_val_test']
    
    return metadata_df

def population_mean_pca(connectivity_tensor, apply_fisher_z=False):
    """
    Compute the population mean connectivity across subjects, then perform PCA using sklearn.

    Args:
        connectivity_tensor: torch.Tensor or np.ndarray of shape (n_subjects, n_edges) containing connectivity values
        apply_fisher_z: If True, apply Fisher z-transform before averaging.

    Returns:
        mean_connectivity: torch.Tensor of shape (n_edges,) containing the population mean.
        scores: torch.Tensor of shape (n_subjects, n_components) - PCA scores (subject projections).
        loadings: torch.Tensor of shape (n_edges, n_components) - PCA components (eigenvectors).
    """
    # Optionally apply Fisher z-transform
    if apply_fisher_z:
        connectivity_tensor = torch.clamp(connectivity_tensor, -1.0, 1.0)
        connectivity_tensor = torch.atanh(connectivity_tensor)
    
    # Compute mean across subjects
    mean_connectivity = np.mean(connectivity_tensor, axis=0)

    # Center the data by subtracting mean
    conn_centered = connectivity_tensor - mean_connectivity

    # Determine n_components for PCA
    n_subjects, n_edges = conn_centered.shape
    n_components_for_pca = min(n_subjects, n_edges)

    pca = PCA(n_components=n_components_for_pca)
    scores = pca.fit_transform(conn_centered)  # shape: (n_subjects, n_components)
    loadings = pca.components_.T               # shape: (n_edges, n_components)

    return mean_connectivity, loadings, scores