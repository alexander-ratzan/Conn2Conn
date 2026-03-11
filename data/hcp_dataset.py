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

VALID_MODALITIES = {"SC", "FC", "SC_r2t"}

# Subset of FreeSurfer features representing whole-brain volumes
FS_VOLUME_COLUMNS = [
    "FS_IntraCranial_Vol",
    "FS_BrainSeg_Vol",
    "FS_BrainSeg_Vol_No_Vent",
    "FS_BrainSeg_Vol_No_Vent_Surf",
    "FS_LCort_GM_Vol",
    "FS_RCort_GM_Vol",
    "FS_TotCort_GM_Vol",
    "FS_SubCort_GM_Vol",
    "FS_Total_GM_Vol",
    "FS_SupraTentorial_Vol",
    "FS_L_WM_Vol",
    "FS_R_WM_Vol",
    "FS_Tot_WM_Vol",
    "FS_Mask_Vol",
    "FS_BrainSegVol_eTIV_Ratio",
    "FS_MaskVol_eTIV_Ratio",
]


def normalize_modality_spec(spec):
    """Normalize a source/target modality spec to a list of modality names."""
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split("+") if part.strip()]
    elif isinstance(spec, (list, tuple)):
        parts = [str(part).strip() for part in spec if str(part).strip()]
    else:
        raise TypeError(f"Unsupported modality spec type: {type(spec)!r}")

    if not parts:
        raise ValueError("Modality spec must include at least one modality.")

    invalid = [part for part in parts if part not in VALID_MODALITIES]
    if invalid:
        raise ValueError(f"Unknown modalities {invalid}. Valid options: {sorted(VALID_MODALITIES)}")
    return parts

class HCP_Base():
    def __init__(self,
    HCP_dir='/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/',
    parcellation='Glasser', hemi='both', source='SC', target='FC', shuffle_seed=0,
    sc_metric_type='sift_invnodevol_radius2_count_connectivity', sc_apply_log1p=True,
    num_pca_components_sc=256, num_pca_components_fc=256,
    cov_sources=None, cov_one_hot_mode="embedding"):
        """
        Load and cache all global HCP data for one fixed experiment data setup.
        The selected source/target modalities are owned by this base instance and
        reused by partitions and models for the full run.
        """
        # Choose parcellation
        self.parcellation = parcellation # Glasser or 4S456Parcels
        self.hemi = hemi # select whether to subset to a specific hemisphere or both
        self.HCP_dir = HCP_dir
        self.source_modalities = normalize_modality_spec(source)
        self.target_modalities = normalize_modality_spec(target)
        if len(self.target_modalities) != 1:
            raise ValueError(
                "Conn2Conn currently supports a single target modality. "
                f"Received target={self.target_modalities}."
            )
        self.source = "+".join(self.source_modalities)
        self.target = self.target_modalities[0]
        self.sc_metric_type = sc_metric_type
        self.sc_apply_log1p = sc_apply_log1p
        self.shuffle_seed = shuffle_seed
        self.num_pca_components_sc = num_pca_components_sc
        self.num_pca_components_fc = num_pca_components_fc
        self.cov_sources = list(cov_sources) if cov_sources is not None else ["fs_all"]
        self.cov_one_hot_mode = cov_one_hot_mode

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
        self.sc_r2t_matrices = (
            self.sc_r2t_matrices[canonical_sc_indices]
            if self.sc_r2t_matrices is not None
            else None
        )
        
        # Compute r x r correlation matrix for each subject's r2t matrix, replace NaNs with 0s in place
        if self.sc_r2t_matrices is None:
            self.sc_r2t_corr_matrices = None
            self.sc_r2t_corr_upper_triangles = None
        else:
            self.sc_r2t_corr_matrices = np.stack(
                [np.nan_to_num(np.corrcoef(mat), nan=0.0) for mat in self.sc_r2t_matrices],
                axis=0,
            )
            tri_indices = np.triu_indices(self.sc_r2t_corr_matrices.shape[1], k=1)
            self.sc_r2t_corr_upper_triangles = self.sc_r2t_corr_matrices[:, tri_indices[0], tri_indices[1]]
        

        self.trainvaltest_partition_indices = {
            "train": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "train"]["subject"].tolist()),
            "val": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "val"]["subject"].tolist()),
            "test": self.subject_indices_from_id(self.all_subject_ids, self.metadata_df[self.metadata_df["train_val_test"] == "test"]["subject"].tolist()),
        }

        self.trainvaltest_partition_ids = {
            "train": self.metadata_df[self.metadata_df["train_val_test"] == "train"]["subject"].tolist(),
            "val": self.metadata_df[self.metadata_df["train_val_test"] == "val"]["subject"].tolist(),
            "test": self.metadata_df[self.metadata_df["train_val_test"] == "test"]["subject"].tolist(),
        }

        # Build aligned FreeSurfer feature matrix and normalize using train split statistics.
        self.fs_feature_columns = [col for col in self.freesurfer_df.columns if col != "subject"]
        self.fs_features_all = self.freesurfer_df[self.fs_feature_columns].to_numpy(dtype=np.float32, copy=True)
        train_indices = self.trainvaltest_partition_indices["train"]
        self.fs_train_mean = self.fs_features_all[train_indices].mean(axis=0)
        self.fs_train_std = self.fs_features_all[train_indices].std(axis=0)
        self.fs_train_std[self.fs_train_std == 0] = 1.0
        self.fs_features_z = (self.fs_features_all - self.fs_train_mean) / self.fs_train_std

        # Precompute FS volume subset (if available)
        available_volume_cols = [col for col in FS_VOLUME_COLUMNS if col in self.freesurfer_df.columns]
        self.fs_volume_columns = available_volume_cols
        if available_volume_cols:
            fs_vol = self.freesurfer_df[available_volume_cols].to_numpy(dtype=np.float32, copy=True)
            self.fs_vol_train_mean = fs_vol[train_indices].mean(axis=0)
            self.fs_vol_train_std = fs_vol[train_indices].std(axis=0)
            self.fs_vol_train_std[self.fs_vol_train_std == 0] = 1.0
            self.fs_volumes_z = (fs_vol - self.fs_vol_train_mean) / self.fs_vol_train_std
        else:
            self.fs_volumes_z = None

        # Precompute one-hot covariate matrix and optional embedding ids.
        if self.covariate_one_hot_tuple:
            cov_one_hot = np.concatenate(self.covariate_one_hot_tuple, axis=1).astype(np.float32)
            self.cov_one_hot_matrix = cov_one_hot
            # Build a compact id per unique one-hot tuple for embedding lookup.
            unique_rows, inverse = np.unique(cov_one_hot, axis=0, return_inverse=True)
            self.cov_one_hot_ids = inverse.astype(np.int64)
            self.cov_one_hot_vocab_size = unique_rows.shape[0]
        else:
            self.cov_one_hot_matrix = None
            self.cov_one_hot_ids = None
            self.cov_one_hot_vocab_size = 0

        # Covariate dimension registry for model construction.
        self.cov_dims = {}
        self.cov_dims["fs_all"] = int(self.fs_features_z.shape[1])
        if self.fs_volumes_z is not None:
            self.cov_dims["fs_volumes"] = int(self.fs_volumes_z.shape[1])
        if self.cov_one_hot_mode == "embedding":
            self.cov_dims["covariate_one_hot_vocab_size"] = int(self.cov_one_hot_vocab_size)
        else:
            if self.cov_one_hot_matrix is not None:
                self.cov_dims["covariate_one_hot"] = int(self.cov_one_hot_matrix.shape[1])
        
        # Compute mean and PCA for training subjects for FC and SC
        self.sc_train_avg,  self.sc_train_loadings, self.sc_train_scores = population_mean_pca(self.sc_upper_triangles[train_indices])
        self.fc_train_avg, self.fc_train_loadings, self.fc_train_scores = population_mean_pca(self.fc_upper_triangles[train_indices])
        self.sc_r2t_corr_train_avg, self.sc_r2t_corr_train_loadings, self.sc_r2t_corr_train_scores = population_mean_pca(self.sc_r2t_corr_upper_triangles[train_indices])

    @staticmethod
    def subject_indices_from_id(subject_list, target_subjects):
        return [i for i, subj in enumerate(subject_list) if subj in target_subjects]

class HCP_Partition(Dataset):
    def __init__(self, base, partition):
        """
        Args:
            base: HCP_Base instance that already owns source/target modality identity
                and global arrays for this run.
            partition: One of ["train", "val", "test"] specifying which split for this dataset.
        """
        self.base = base
        self.partition = partition
        self.indices = base.trainvaltest_partition_indices[partition] # global idx list like: [2, 4, 15, 19, 21, 24...]
        self.ids = base.trainvaltest_partition_ids[partition] # id list like: ["100206", "100207", "100208", "100209", "100210", "100211"]
        self.ids_to_indices = dict  (zip(self.ids, self.indices))

        self.source_modalities = list(base.source_modalities)
        self.target = base.target

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sc_upper_triangles = torch.tensor(base.sc_upper_triangles, dtype=torch.float32, device=self.device)
        self.fc_upper_triangles = torch.tensor(base.fc_upper_triangles, dtype=torch.float32, device=self.device)
        self.sc_r2t_corr_upper_triangles = (
            torch.tensor(base.sc_r2t_corr_upper_triangles, dtype=torch.float32, device=self.device)
            if base.sc_r2t_corr_upper_triangles is not None
            else None
        )
        # Prebuild covariate tensors for requested sources.
        self.cov_sources = list(base.cov_sources)
        self.cov_tensors = {}
        if "fs_all" in self.cov_sources:
            self.cov_tensors["fs_all"] = torch.tensor(base.fs_features_z, dtype=torch.float32, device=self.device)
        if "fs_volumes" in self.cov_sources:
            if base.fs_volumes_z is None:
                raise ValueError("fs_volumes requested but FS volume columns are unavailable.")
            self.cov_tensors["fs_volumes"] = torch.tensor(base.fs_volumes_z, dtype=torch.float32, device=self.device)
        if "covariate_one_hot" in self.cov_sources:
            if base.cov_one_hot_mode == "embedding":
                if base.cov_one_hot_ids is None:
                    raise ValueError("covariate_one_hot requested but IDs are unavailable.")
                self.cov_tensors["covariate_one_hot"] = torch.tensor(base.cov_one_hot_ids, dtype=torch.long, device=self.device)
            else:
                if base.cov_one_hot_matrix is None:
                    raise ValueError("covariate_one_hot requested but matrix is unavailable.")
                self.cov_tensors["covariate_one_hot"] = torch.tensor(base.cov_one_hot_matrix, dtype=torch.float32, device=self.device)
    
    def _get_modality_tensor(self, modality, global_idx):
        if modality == "SC":
            return self.sc_upper_triangles[global_idx]
        if modality == "FC":
            return self.fc_upper_triangles[global_idx]
        if modality == "SC_r2t":
            if self.sc_r2t_corr_upper_triangles is None:
                raise ValueError("SC_r2t requested but sc_r2t correlation matrices are unavailable.")
            return self.sc_r2t_corr_upper_triangles[global_idx]
        raise ValueError(f"Unknown modality: {modality}")

    def __getitem__(self, idx):
        """
        Return one subject sample for this split using the base-fixed source/target
        modality setup.
        """
        global_idx = self.indices[idx] # retruns index of subject in global subject list
        source_modalities = {
            modality: self._get_modality_tensor(modality, global_idx)
            for modality in self.source_modalities
        }
        target_data = self._get_modality_tensor(self.target, global_idx)
        cov_data = {key: tensor[global_idx] for key, tensor in self.cov_tensors.items()}

        if len(self.source_modalities) == 1:
            model_input = source_modalities[self.source_modalities[0]]
        else:
            model_input = source_modalities

        return {
            "x": model_input,
            "x_modalities": source_modalities,
            "y": target_data,
            "cov": cov_data,
            "subject_id": self.ids[idx],
        }
    
    def __len__(self):
        return len(self.indices)
