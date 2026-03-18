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

DEFAULT_CONN2CONN_CACHE_ROOT = "/scratch/asr655/neuroinformatics/Conn2Conn_data"


def _sanitize_token(token):
    return str(token).replace("/", "_").replace(" ", "_").replace("+", "p")


def _write_npy_cache(cache_dir, arrays):
    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in arrays.items():
        if arr is None:
            continue
        np.save(os.path.join(cache_dir, f"{name}.npy"), arr)

def _read_required_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing precomputed cache file: {path}")
    return np.load(path)


def _cache_dir_fc(cache_root, parcellation, hemi):
    return os.path.join(
        cache_root,
        "fc",
        f"parc-{_sanitize_token(parcellation)}_hemi-{_sanitize_token(hemi)}",
    )


def _cache_dir_sc(cache_root, parcellation, hemi, metric_type, apply_log1p):
    return os.path.join(
        cache_root,
        "sc",
        (
            f"parc-{_sanitize_token(parcellation)}_hemi-{_sanitize_token(hemi)}"
            f"_metric-{_sanitize_token(metric_type)}_log1p-{int(bool(apply_log1p))}"
        ),
    )


def _cache_dir_parcel_node_features(cache_root, parcellation, hemi, volume_feature_type, centroid_feature_type):
    return os.path.join(
        cache_root,
        "parcel_node_features",
        (
            f"parc-{_sanitize_token(parcellation)}_hemi-{_sanitize_token(hemi)}"
            f"_vol-{_sanitize_token(volume_feature_type)}_cent-{_sanitize_token(centroid_feature_type)}"
        ),
    )


def load_fc_precomputed(
    parcellation='Glasser',
    hemi='both',
    cache_root=DEFAULT_CONN2CONN_CACHE_ROOT,
):
    """Load FC arrays from precomputed npy cache."""
    cache_dir = _cache_dir_fc(cache_root, parcellation, hemi)
    subject_ids = _read_required_npy(os.path.join(cache_dir, "subject_ids.npy")).astype(np.int64).tolist()
    fc_matrices = _read_required_npy(os.path.join(cache_dir, "matrices.npy")).astype(np.float32)
    tri_path = os.path.join(cache_dir, "upper_triangles.npy")
    if os.path.exists(tri_path):
        fc_triangles = np.load(tri_path).astype(np.float32)
    else:
        tri_indices = np.triu_indices(fc_matrices.shape[1], k=1)
        fc_triangles = fc_matrices[:, tri_indices[0], tri_indices[1]].astype(np.float32)
    return subject_ids, fc_matrices, fc_triangles


def load_sc_precomputed(
    parcellation='Glasser',
    hemi='both',
    metric_type='sift_invnodevol_radius2_count_connectivity',
    apply_log1p=False,
    cache_root=DEFAULT_CONN2CONN_CACHE_ROOT,
):
    """Load SC arrays (and optional r2t) from precomputed npy cache."""
    cache_dir = _cache_dir_sc(cache_root, parcellation, hemi, metric_type, apply_log1p)
    subject_ids = _read_required_npy(os.path.join(cache_dir, "subject_ids.npy")).astype(np.int64).tolist()
    sc_matrices = _read_required_npy(os.path.join(cache_dir, "matrices.npy")).astype(np.float32)
    tri_path = os.path.join(cache_dir, "upper_triangles.npy")
    if os.path.exists(tri_path):
        sc_triangles = np.load(tri_path).astype(np.float32)
    else:
        tri_indices = np.triu_indices(sc_matrices.shape[1], k=1)
        sc_triangles = sc_matrices[:, tri_indices[0], tri_indices[1]].astype(np.float32)
    r2t_path = os.path.join(cache_dir, "r2t_matrices.npy")
    sc_r2t_matrices = np.load(r2t_path).astype(np.float32) if os.path.exists(r2t_path) else None
    return subject_ids, sc_matrices, sc_triangles, sc_r2t_matrices


def load_parcel_volume_centroids_precomputed(
    parcellation='Glasser',
    hemi='both',
    volume_feature_type='volume_mm3',
    centroid_feature_type='centroid_mm',
    cache_root=DEFAULT_CONN2CONN_CACHE_ROOT,
):
    """Load cached per-node parcel features from precomputed npy cache."""
    cache_dir = _cache_dir_parcel_node_features(
        cache_root,
        parcellation,
        hemi,
        volume_feature_type,
        centroid_feature_type,
    )
    subject_ids = _read_required_npy(os.path.join(cache_dir, "subject_ids.npy")).astype(np.int64).tolist()
    parcel_volume = _read_required_npy(os.path.join(cache_dir, "parcel_volume.npy")).astype(np.float32)
    parcel_centroids = _read_required_npy(os.path.join(cache_dir, "parcel_centroids.npy")).astype(np.float32)
    node_path = os.path.join(cache_dir, "parcel_node_features.npy")
    if os.path.exists(node_path):
        parcel_node_features = np.load(node_path).astype(np.float32)
    else:
        parcel_node_features = np.concatenate([parcel_volume[..., None], parcel_centroids], axis=-1).astype(np.float32)
    return subject_ids, parcel_volume, parcel_centroids, parcel_node_features


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

def load_fc(
    parcellation='Glasser',
    hemi='both',
    HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
    n_jobs=None,
    write_cache=False,
    cache_root=DEFAULT_CONN2CONN_CACHE_ROOT,
):
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
    else: #elif hemi == 'both': roi_mask = roi_df['hemisphere'].str.contains('L') | roi_df['hemisphere'].str.contains('R')
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
    
    if write_cache:
        cache_dir = _cache_dir_fc(cache_root, parcellation, hemi)
        _write_npy_cache(
            cache_dir,
            {
                "subject_ids": np.asarray(subject_ids, dtype=np.int64),
                "matrices": fc_matrices.astype(np.float32),
                "upper_triangles": fc_triangles.astype(np.float32),
            },
        )

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
            apply_log1p=False, n_jobs=None, write_cache=False,
            cache_root=DEFAULT_CONN2CONN_CACHE_ROOT):
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
    else: #elif hemi == 'both': roi_mask = roi_df['hemisphere'].str.contains('L') | roi_df['hemisphere'].str.contains('R')
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

    if write_cache:
        cache_dir = _cache_dir_sc(cache_root, parcellation, hemi, metric_type, apply_log1p)
        _write_npy_cache(
            cache_dir,
            {
                "subject_ids": np.asarray(subject_ids, dtype=np.int64),
                "matrices": sc_matrices.astype(np.float32),
                "upper_triangles": sc_triangles.astype(np.float32),
                "r2t_matrices": None if sc_r2t_matrices is None else sc_r2t_matrices.astype(np.float32),
            },
        )

    return subject_ids, sc_matrices, sc_triangles, sc_r2t_matrices


def _resolve_volume_column(volume_feature_type: str) -> str:
    if volume_feature_type == "volume_mm3":
        return "volume_mm3"
    if volume_feature_type in {"normalized", "normalized_voxel_count"}:
        return "normalized_voxel_count"
    raise ValueError(
        f"Unknown volume_feature_type='{volume_feature_type}'. "
        "Choose from {'volume_mm3', 'normalized'}."
    )


def _resolve_centroid_columns(centroid_feature_type: str):
    if centroid_feature_type == "centroid_mm":
        return ["centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]
    if centroid_feature_type == "medoid":
        return ["medoid_x_mm", "medoid_y_mm", "medoid_z_mm"]
    raise ValueError(
        f"Unknown centroid_feature_type='{centroid_feature_type}'. "
        "Choose from {'centroid_mm', 'medoid'}."
    )


def _load_single_volume_centroid_file(args):
    """Helper to load one subject's per-parcel volume/centroid CSV."""
    HCP_dir, subj_folder, parcellation, volume_feature_type, centroid_feature_type = args
    subject_id = subj_folder.replace("sub-", "")

    csv_path = os.path.join(
        HCP_dir,
        "HCP1200_DTI/qsirecon",
        subj_folder,
        "anat",
        "T1",
        f"{subj_folder}_space-T1w_seg-{parcellation}_dseg_volumes_centroids.csv",
    )
    if not os.path.exists(csv_path):
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        if "atlas" in df.columns:
            df = df[df["atlas"] == parcellation].copy()
        if "label" in df.columns:
            df = df.sort_values("label")

        vol_col = _resolve_volume_column(volume_feature_type)
        centroid_cols = _resolve_centroid_columns(centroid_feature_type)

        missing = [c for c in [vol_col, *centroid_cols] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in {csv_path}."
            )

        volume_vals = df[vol_col].to_numpy(dtype=np.float32)
        centroid_vals = df[centroid_cols].to_numpy(dtype=np.float32)
        return subject_id, volume_vals, centroid_vals
    except Exception as e:
        print(f"[load_parcel_volume_centroids] Error loading {csv_path}: {e}. Skipping.")
        return None, None, None


def load_parcel_volume_centroids(
    parcellation='Glasser',
    hemi='both',
    HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
    volume_feature_type='volume_mm3',
    centroid_feature_type='centroid_mm',
    n_jobs=None,
    write_cache=False,
    cache_root=DEFAULT_CONN2CONN_CACHE_ROOT,
):
    """
    Load per-subject per-node volume and centroid features from T1 segmentation CSVs.

    Args:
        parcellation: 'Glasser' or '4S456Parcels'
        hemi: 'left' | 'right' | 'both'
        volume_feature_type:
            - 'volume_mm3' (default)
            - 'normalized' (maps to normalized_voxel_count)
        centroid_feature_type:
            - 'centroid_mm' (default; centroid_x/y/z_mm)
            - 'medoid' (medoid_x/y/z_mm)
        n_jobs: number of parallel workers

    Returns:
        subject_ids: list[int]
        parcel_volume: np.ndarray [n_subjects, n_roi]
        parcel_centroids: np.ndarray [n_subjects, n_roi, 3]
        parcel_node_features: np.ndarray [n_subjects, n_roi, 4]
            concatenation of [volume, centroid_xyz]
    """
    subjects_dir = os.path.join(HCP_dir, "HCP1200_fMRI/xcpd-0-9-1/")
    subject_folders = [d for d in os.listdir(subjects_dir) if d.startswith("sub-")]
    subject_folders = sorted(subject_folders)

    args_list = [
        (HCP_dir, subj_folder, parcellation, volume_feature_type, centroid_feature_type)
        for subj_folder in subject_folders
    ]
    n_jobs = min(len(subject_folders), os.cpu_count() or 4) if n_jobs is None else n_jobs

    # Hemisphere mask from atlas metadata (same indexing convention as FC/SC loaders).
    roi_df = pd.read_csv(
        f'/scratch/asr655/neuroinformatics/Conn2Conn/data/atlas_info/{parcellation}_dseg_reformatted.csv'
    )
    if hemi == 'left':
        roi_mask = roi_df['hemisphere'].str.contains('L')
    elif hemi == 'right':
        roi_mask = roi_df['hemisphere'].str.contains('R')
    else:
        roi_mask = np.ones(len(roi_df), dtype=bool)
    roi_indices = np.where(roi_mask)[0]

    subject_ids = []
    volume_list = []
    centroid_list = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(_load_single_volume_centroid_file, args_list)

    for subject_id, volume_vals, centroid_vals in results:
        if subject_id is None:
            continue
        if volume_vals is None or centroid_vals is None:
            continue
        if volume_vals.shape[0] <= roi_indices.max() or centroid_vals.shape[0] <= roi_indices.max():
            print(
                f"[load_parcel_volume_centroids] Subject {subject_id}: feature rows "
                f"({volume_vals.shape[0]}) do not match expected ROI indexing "
                f"(max idx {roi_indices.max()}). Skipping."
            )
            continue
        volume_vals = volume_vals[roi_indices]
        centroid_vals = centroid_vals[roi_indices, :]
        subject_ids.append(int(subject_id))
        volume_list.append(volume_vals)
        centroid_list.append(centroid_vals)

    if len(subject_ids) == 0:
        raise RuntimeError(
            "No subject volume/centroid files found or loaded. "
            "Check T1 CSV generation and parcellation."
        )

    parcel_volume = np.stack(volume_list, axis=0).astype(np.float32)
    parcel_centroids = np.stack(centroid_list, axis=0).astype(np.float32)
    parcel_node_features = np.concatenate(
        [parcel_volume[..., None], parcel_centroids], axis=-1
    ).astype(np.float32)

    if write_cache:
        cache_dir = _cache_dir_parcel_node_features(
            cache_root,
            parcellation,
            hemi,
            volume_feature_type,
            centroid_feature_type,
        )
        _write_npy_cache(
            cache_dir,
            {
                "subject_ids": np.asarray(subject_ids, dtype=np.int64),
                "parcel_volume": parcel_volume.astype(np.float32),
                "parcel_centroids": parcel_centroids.astype(np.float32),
                "parcel_node_features": parcel_node_features.astype(np.float32),
            },
        )

    return subject_ids, parcel_volume, parcel_centroids, parcel_node_features

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

def load_metadata(shuffle_seed=0, rare_race_eth_threshold=10):
    """
    Loads and merges metadata from participants.tsv and HCP1200_RESTRICTED.csv.

    Age is returned as a raw continuous float array (z-scoring is deferred to HCP_Base
    so it is computed on the training split only).  Sex and race_eth are one-hot encoded.
    Race/ethnicity categories with fewer than ``rare_race_eth_threshold`` training subjects
    are collapsed into an "Other" bucket, guaranteeing val/test coverage.

    Args:
        shuffle_seed: Random seed for train/val/test split generation.
                      If shuffle_seed=0, uses the original split from participants.tsv.
                      If shuffle_seed != 0, generates a new family-preserving split using
                      that seed.
        rare_race_eth_threshold: Race/ethnicity categories whose training-set count is
                      below this value are merged into "Other".  Defaults to 10.

    Returns:
        metadata_df: DataFrame with columns:
            - subject, train_val_test, age, sex, Race_Ethnicity,
              Family_Relation, Family_ID
        covariate_arrays: dict with keys:
            - "age"      : float32 array  (N,)   — continuous age in years
            - "sex"      : float32 array  (N, 2) — one-hot [F, M]
            - "race_eth" : float32 array  (N, k) — collapsed race/eth one-hot
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

    cols = ['subject', 'train_val_test', 'age', 'sex', 'Race_Ethnicity', 'Family_Relation', 'Family_ID']
    metadata_df = metadata_df[cols]

    # --- Continuous age ---
    age_np = metadata_df['age'].to_numpy(dtype=np.float32)

    # --- Sex one-hot (always exactly 2 categories: F, M) ---
    sex_oh_np = pd.get_dummies(metadata_df['sex'], dtype=np.float32).to_numpy()

    # --- Race/ethnicity one-hot with rare-category collapsing ---
    # Count per category in training subjects only; merge rare ones into "Other"
    # so that every category seen in val/test is guaranteed present in training.
    train_mask = metadata_df['train_val_test'] == 'train'
    train_counts = metadata_df.loc[train_mask, 'Race_Ethnicity'].value_counts()
    rare_categories = set(train_counts[train_counts < rare_race_eth_threshold].index)
    # Also collapse categories entirely absent from training.
    all_train_cats = set(train_counts.index)
    all_cats = set(metadata_df['Race_Ethnicity'].unique())
    rare_categories |= (all_cats - all_train_cats)
    race_eth_collapsed = metadata_df['Race_Ethnicity'].apply(
        lambda x: 'Other' if x in rare_categories else x
    )
    race_eth_oh_np = pd.get_dummies(race_eth_collapsed, dtype=np.float32).to_numpy()

    covariate_arrays = {
        "age":      age_np,
        "sex":      sex_oh_np,
        "race_eth": race_eth_oh_np,
    }

    return metadata_df, covariate_arrays

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
