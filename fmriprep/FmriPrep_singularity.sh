#!/bin/bash
#
# The fMRIPrep script is meant to streamline the preprocessing of fMRI dataset.
# The input is a bids dataset 
# The output is preprocessed data set ready for first level analysis. 
# The script can be run on the complete dataset or every # subjects. Make sure all subjects are ran on the same fMRIPrep version.
# This is a bash script and should be ran from the HPC terminal (HPC desktop/shell/VScode)
# This script uses full path, hence, you do not have to be in a spesific directory to run it. 
# Tip, create a new folder and run the script from it as the script produce (2*number of subject) log files
#
# Inorder for fMRIPrep to work cleanly without the need to create/load conda enviroment it is packed into a containr named singularity
# The singularity is packed in an img file (no need to load a singulrity module).
# This script loads the fMRIprep img file and automatically runs all the preprocessing phases on the bids compatible dataset.
# In order to run it, you will have to validate the dataset using
# http://bids-standard.github.io/bids-validator/
# or add the --skip-bids-validation decoration. 
#
# Once your dataset is ready (bids validated/decided to ignore) 
# 1. change the name of the job (what ever you feel like) on line 25
# 2. Change the number of participants on line 26 and change email addres on line 36
# 3. add a list of subjects
# 4. adjust directories path
# 5. From the HPC terminal run using the command "sbatch FmriPrep_singularity.sh"
#
#SBATCH --j RAID_11to62 # job name
#SBATCH --array=1-35 # number of participants as range starting at 1 (i.e., for 5 participants: 1-5)
#SBATCH --time=48:00:00 # HPC will give you this amount of time to run the process. This is usually enough time
#SBATCH -n 1 # how many nodes you are asking. This is running each subject on a differnt node so 1 is enough
#SBATCH --cpus-per-task=4 # How many CPUs. This is enough cpus no need to change
#SBATCH --mem-per-cpu=8G # How much memory per CPU. This is enough memory no need to change

# resouce you are using are nodes * CPUs * memory - if you go above 120 per subject you will have to wait a lot of time to get an opening
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a.out # this will give you the list of commands and there results (success/failure). If the run fails here you will get the spesifcs
#SBATCH -e %x-%A-%a.err # this will give you a short file with what errors were during the execution
#SBATCH --mail-user=chelsea.xu@yale.edu # replace with your email!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --mail-type=ALL
# ------------------------------------------

# enter subject list with only space between them and the "sub-" prefix (i.e. sub-10 sub-11)
SUBJ=(sub-11 sub-12 sub-13 sub-15 sub-16 sub-17 sub-19 sub-20 sub-21 sub-22 sub-24 sub-25 sub-27 sub-28 sub-29 sub-30 sub-31 sub-32 sub-36 sub-39 sub-40 sub-41 sub-42 sub-43 sub-45 sub-46 sub-47 sub-48 sub-50 sub-51 sub-55 sub-56 sub-57 sub-61 sub-62)

# adjust directories path based on the bids directories
BASE_DIR="/gpfs/gibbs/pi/levy_ifat/Chelsea/RAID" # not really used, not sure why is it here
BIDS_DIR="/gpfs/gibbs/pi/levy_ifat/Chelsea/RAID/R_A_ID_BIDS" # Location of the bids folder.
DERIVS_DIR="derivatives" # the derivatives folder should be inside the bids folder.
WORK_DIR="/home/cyx3/scratch60/work" # enter working directory here - preferably on scratch60.
# If you want to rerun fMRIprep clean the working directory before

mkdir -p $HOME/.cache/templateflow
mkdir -p ${BIDS_DIR}/${DERIVS_DIR}
mkdir -p ${BIDS_DIR}/${DERIVS_DIR}/freesurfer-6.0.1
ln -s    ${BIDS_DIR}/${DERIVS_DIR}/freesurfer-6.0.1 ${BIDS_DIR}/${DERIVS_DIR}/freesurfer

# this is loading the license to run freesurfer
export SINGULARITYENV_FS_LICENSE=/gpfs/gibbs/pi/levy_ifat/shared/licenseFreeSurfer.txt # freesurfer license file

# this create a folder for fMRIprep. If fMRIprep fails to run clean this folder 
# cd ~/.cache/templateflow/
# rm *
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"

# Load the fMRIprep img
SINGULARITY_CMD="singularity run --cleanenv -B $HOME/.cache/templateflow:/templateflow -B ${WORK_DIR}:/work /gpfs/gibbs/pi/levy_ifat/shared/fmriPrep/fmriprep-21.0.1.simg"

# this is where the magic starts
echo Starting ${SUBJ[$SLURM_ARRAY_TASK_ID-1]}

# this is the line that runs the code. If you want to use --skip-bids-validation, enter it as part of the next line (before the --output-space decoration).
cmd="${SINGULARITY_CMD} ${BIDS_DIR} ${BIDS_DIR}/${DERIVS_DIR} participant --participant-label ${SUBJ[$SLURM_ARRAY_TASK_ID-1]} -w /work/ -vv --omp-nthreads 8 --nthreads 12 --mem_mb 30000 --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 --cifti-output"
      # --use-aroma"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
