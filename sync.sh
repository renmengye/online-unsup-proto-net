MACHINE=$1
SRC=$PWD
DEST="$(dirname "$SRC")"
rsync -av --progress \
  --exclude '.git' \
  --exclude '*.swp' \
  --exclude '__pycache__' \
  --exclude 'output' \
  --exclude 'results' \
  --exclude 'allplots' \
  --exclude '*.png' \
  $SRC $MACHINE:$DEST
