echo $2
folder=$1
addr="https://nyu.databrary.org/volume/564/slot/$2/zip/true"
output="databrary-564-$2.zip"
echo $addr
wget --referer="https://nyu.databrary.org/api/user/login" --cookies=on --keep-session-cookies --load-cookies=cookie.txt $addr -O $output -P $folder