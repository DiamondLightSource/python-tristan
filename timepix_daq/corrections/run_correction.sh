#!/bin/bash
# Pass main experiment directory as command line argument
wd=$1
echo "Experiment directory: $wd"

timepix=$PWD

# Find location of xml file for this experiment from .log file
for file in $(find $wd -name "*.log"); do 
    echo "Log file found: $file"
    xml=$(grep 'xml file:' $file | awk '{print $NF}')
    #echo $xml 
done
# XML file location
echo "xml file location: $xml"

# Execute python script
module load python/3
module load hdf5

# Go through the subdirectories one by one
for file in $(find $wd -name "*_vds.h5"); do
    echo $file
    filedir=$(dirname $file)
    dir=$(dirname $wd)/processing/correct_nexus/$(basename $wd)/$(basename $filedir)
    # Use this only to test that it does its job. Correct one is commented out above
    #dir=$(dirname $wd)/processing/nf/correct_nexus/$(basename $wd)/$(basename $filedir)
    # Create directory if it doesn't exist
    if ! [ -d $dir ]; then
        echo "Directory does not exist"
        echo "Creating directory now"
        mkdir -p $dir
    else
        echo "Directory alrady exists"
        echo "Writing to directory $dir"
    fi
    # Output file name
    name=$(echo $(basename $file) | awk -F "_vds" '{printf $1}')
    nxs="${name}.nxs"
    # Run make_new_nxs.py
    python $timepix/make_new_NXS.py $dir/$nxs $file $xml
done


