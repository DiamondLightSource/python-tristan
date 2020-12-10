#!/bin/bash
# Pass main experiment directory as command line argument
wd=$1
echo "Experiment directory: $wd"

# # Create directory in processing/correct_nexus
# dir=$(dirname $edir)/processing/correct_nexus/$(basename $edir)
# if ! [ -d $dir ]; then
#     echo "Directory does not exist."
#     echo "Creating $dir now."
#     mkdir -p dir    #Not sure this is right
# fi

# Find location of xml file for this experiment from .log file
for file in $(find $wd -name "*.log"); do 
    xml=$(grep 'xml file: ' $file | awk '{print $NF}')
    echo $xml 
done

# Execute python script
module load python/3
module load hdf5

# Go through the subdirectories one by one
for file in $(find $wd -name "*_vds.h5"); do
    echo $file
    filedir=$(dirname $file)
    #dir=$(dirname $wd)/processing/correct_nexus/$(basename $wd)/$(basename $filedir)
    # FIXME use this only to test that it does its job. Correct one is commented out above
    dir=$(dirname $wd)/processing/nf/correct_nexus/$(basename $wd)/$(basename $filedir)
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
    outfile=$(echo $(basename $file) | awk -F "_vds" '{printf $1}')
    nxs="${name}.nxs"
    # Call python script that opens nexus file and gets count_time and comment
    # Run make_new_nxs.py
    # python make_new_nexus.py $dir/$nxs $file $xml
done


# Call python script that opens nexus file and gets count_time and comment
# Get location of vds.h5 and meta.h5 files
# Run make_new_nxs.py