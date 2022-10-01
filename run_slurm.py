import argparse
import subprocess
import time


def main():
  fname = 'submit_guppy.sh'
  with open(fname, 'r') as f:
    alltext = f.read()

  assert args.id is not None
  alltext = alltext.replace('{ID}', args.id)
  alltext = alltext.replace('{PROG}', args.prog)
  alltext = alltext.replace('{NGPU}', str(args.ngpu))
  alltext = alltext.replace('{NODES}', str(args.nodes))
  alltext = alltext.replace('{MEM}', args.mem)
  alltext = alltext.replace('{NCPU}', str(args.ncpu))
  alltext = alltext.replace('{PARTITION}', str(args.partition))

  if args.qos is not None:
    alltext = alltext.replace('{QOS}', args.qos)
    if args.qos == "deadline":
      alltext = alltext.replace('{ACCOUNT}', 'deadline')
    else:
      alltext = alltext.replace('#SBATCH --account={ACCOUNT}\n', '')
  else:
    alltext = alltext.replace('#SBATCH --account={ACCOUNT}\n', '')
    alltext = alltext.replace('#SBATCH --qos={QOS}', '')

  if args.exclude is not None:
    alltext = alltext.replace('{EXCLUDE}', args.exclude)
  else:
    alltext = alltext.replace('#SBATCH --exclude={EXCLUDE}', '')

  print(alltext)

  tmp_filename = 'submit_guppy_{}.sh'.format(int(time.time()))
  with open(tmp_filename, 'w') as f:
    f.write(alltext)

  call = subprocess.run(["sbatch", tmp_filename], stdout=subprocess.PIPE)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--id', type=str, default=None)
  parser.add_argument('--prog', type=str, default="")
  parser.add_argument('--mem', type=str, default="128G")
  parser.add_argument('--qos', type=str, default=None)
  parser.add_argument('--ngpu', type=int, default=4)
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--ncpu', type=int, default=20)
  parser.add_argument('--partition', type=str, default="gpu")
  parser.add_argument('--exclude', type=str, default=None)
  args = parser.parse_args()
  main()
