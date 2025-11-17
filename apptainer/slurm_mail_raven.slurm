#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./joblogs/job_test.out.%j
#SBATCH -e ./joblogs/job_test.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_apu_slurm
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#
# --- default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100000
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
module load gcc/14 openmpi/5.0 rocm/6.3
module load apptainer


srun apptainer exec --bind /ptmp/schmidtjo/11_emc/ /ptmp/schmidtjo/apptainer/pymritools.sif simulation_emc_mese -c /ptmp/schmidtjo/11_emc/emc_settings.json
