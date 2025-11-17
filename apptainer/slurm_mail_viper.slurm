#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./joblogs/job_test.out.%j
#SBATCH -e ./joblogs/job_test.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J emc_sim_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 APUs on a full node ---
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=48
# #SBATCH --mem=240000
#
#SBATCH --mail-type=END
#SBATCH --mail-user=jschmidt@cbs.mpg.de
# Wall clock limit
#SBATCH --time=18:00:00

module purge
module load intel/21.2.0 impi/2021.2 cuda/11.2
module load apptainer

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun apptainer exec --bind /ptmp/schmidtjo/emc_database_generation/ /ptmp/schmidtjo/apptainer/pymritools.sif simulation_emc_mese -c /ptmp/schmidtjo/emc_database_generation/emc_settings.json
