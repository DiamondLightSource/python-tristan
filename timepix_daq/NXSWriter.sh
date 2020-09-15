# Pass the arguments for the python script
# Input arguments order: vdsfile, xmlfile, exposure_time, comments
# Last argument is optional


vds_file=$1
xml_file=$2
exposure_time=$3
comments=${4:-}

echo "vds: $vds_file"
echo "xml: $xml_file"
echo "exposure_time:$exposure_time"
#echo "msg: $comments"

if [ -z "$comments" ]
then
    echo "There are no comment messages"
else
    echo "msg: $comments"
fi
# Execute python script
module load python/3
module load hdf5

if [ -z "$comments" ]
then
    python /dls_sw/i19-2/software/user.scripts/AndyAlexander/makeNXS.py $vds_file $xml_file $exposure_time
else
    python /dls_sw/i19-2/software/user.scripts/AndyAlexander/makeNXS.py $vds_file $xml_file $exposure_time "$comments"
fi
