#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jvt22 # required to send email notifcations - please replace <your_username> with your college login name or email address
#PBS -l walltime=20:00:00
export PATH=/vol/bitbucket/jvt22/myvenv/bin:$PATH
source activate
# source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
# TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python plot.py