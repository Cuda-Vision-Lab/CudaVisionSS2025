#URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
ZIP_FILE=./data/afhq_v2.zip
mkdir -p "$(dirname "$0")/data"
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d "$(dirname "$0")/data"
rm $ZIP_FILE