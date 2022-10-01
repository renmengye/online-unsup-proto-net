source setup_environ.sh
FOLDER=$DATA_DIR/say-cam
mkdir -p $FOLDER
xargs -0 -n 1 bash download.sh $FOLDER < <(tr \\n \\0 <say_cam_id_list.txt)