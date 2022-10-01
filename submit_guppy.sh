#!/bin/bash
#
#SBATCH --job-name={ID}
#SBATCH --output=logs/{ID}.txt
#SBATCH --error=logs/{ID}_stderr.txt
#
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={NGPU}
#SBATCH --cpus-per-task={NCPU}
#SBATCH --mem={MEM}
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:{NGPU}
#SBATCH --account={ACCOUNT}
#SBATCH --qos={QOS}
#SBATCH --exclude={EXCLUDE}

# Install hostlist from "pip install python-hostlist"

echo "nodelist" ${SLURM_NODELIST}
hosts=$(hostlist -e ${SLURM_NODELIST})
hostfile=""
count=0
for host in $hosts
do
  if [ -z "$hostfile" ]; then
    hostfile="$host:{NGPU}"
  else
    hostfile="$hostfile,$host:{NGPU}"
  fi
  let count=count+1
done
if [ $count == 1 ]; then
  hostfile="localhost:{NGPU}"
fi
echo "host" $hostfile
let np=count*{NGPU}
echo "np" $np
horovodrun -np $np -H $hostfile \
  --mpi-args="--oversubscribe" \
  {PROG}