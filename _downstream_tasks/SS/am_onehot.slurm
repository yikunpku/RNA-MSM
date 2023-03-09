#!/bin/bash
#SBATCH --job-name=torch_bert
#SBATCH --partition=cu-1
##SBATCH --nodelist=gpu10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:1
#SBATCH --error=%j.err
#SBATCH --output=%j.out
##SBATCH --export=ALL
l=`whoami`
CURDIR=`pwd`
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE">>$CURDIR/nodelist.$SLURM_JOB_ID
done
echo $SLURM_NPROCS
echo "process will start at:"
date
module load cuda/cuda-10.2
module load pytorch/pytorch-gpu-1.10.0-py37
cd $SLURM_SUBMIT_DIR
#module load biopython/1.76-py37
#cp -r ./data /tmp
#echo "cp finished"
./am_onehot_run.sh
#module unload cuda/cuda-10.2
#module unload pytorch/pytorch-gpu-1.10.0-py37
#module unload biopython/1.76-py37

echo "++++++++++++++++++++++++++++++++++++++++"
echo "processs will sleep 30s"
sleep 30
echo "process end at : "
date



