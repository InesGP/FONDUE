#!/bin/bash
#SBATCH --job-name=raagav_fondue_test
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-12:00
#SBATCH --account=rrg-glatard
#SBATCH --output=/home/rprasann/scratch/test_fastsurfer_1.1.0/embeddings_project/slurm/%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raagav.pras@gmail.com

module load apptainer/1.2.4


apptainer exec --writable-tmpfs \
     --bind /home/rprasann/scratch/test_fastsurfer_1.1.0/Synthmorph/binds:/binds\
     /home/rprasann/scratch/test_fastsurfer_1.1.0/FONDUE/fondue_no_conda.sif \
     /bin/bash -c "cd /FONDUE && python3 fondue_eval.py --in_name /binds/sub-0025531.nii.gz --keep_dims True --intensity_range_mode 0 --no_cuda"

#apptainer exec --writable-tmpfs \
      #--bind /home/rprasann/scratch/test_fastsurfer_1.1.0/Synthmorph/embeddings:/voxelmorph/embeddings\
      #--bind /home/rprasann/scratch/test_fastsurfer_1.1.0/Synthmorph/binds:/voxelmorph/binds\
      #${SINGULARITY_IMAGE} \
      #python3 /voxelmorph/binds/register.py --moving /voxelmorph/binds/sub-0025531.nii.gz\
      #--fixed /voxelmorph/binds/norm-average_mni305.mgz --model /voxelmorph/binds/brains-dice-vel-0.5-res-16-256f.h5\



