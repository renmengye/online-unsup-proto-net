source setup_environ.sh
DIR=$PWD
cd $DATA_DIR/say-cam
for f in *.zip;
do
  echo $f
  unzip $f
done;
# rm *.zip
cd $DIR
