source /ext3/miniforge3/bin/activate
conda activate /ext3/miniforge3/envs/kraken_env
export PYTHONNOUSERSITE=True

# 5) avoid quota issues by forcing runtime/cache/config to scratch
export SCR_BASE=/scratch/ans9868/.jupyter_scratch
mkdir -p $SCR_BASE/{home,cache,config,conda,tmp,jupyter,runtime,ipython,matplotlib}
export HOME=$SCR_BASE/home
export XDG_CACHE_HOME=$SCR_BASE/cache
export XDG_CONFIG_HOME=$SCR_BASE/config
export CONDARC=$SCR_BASE/conda/.condarc
export TMPDIR=$SCR_BASE/tmp
export JUPYTER_CONFIG_DIR=$SCR_BASE/jupyter
export JUPYTER_DATA_DIR=$SCR_BASE/jupyter
export JUPYTER_RUNTIME_DIR=$SCR_BASE/runtime
export IPYTHONDIR=$SCR_BASE/ipython
export MPLCONFIGDIR=$SCR_BASE/matplotlib

