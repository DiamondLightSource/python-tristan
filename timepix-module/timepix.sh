# Do NOT load modules in this file!
# load them in the main module file instead

# Though the above warning is usually applicable, we ignore it here.
# If we load the modules properly in the main module file, there seems
# to be some conflict that results in the python/3 conda environment
# not getting set up properly.
module load hdf5
module load i19
module load python/3

export TIMEPIX_I19=/dls_sw/i19/scripts/BeamlineScripts/timepix-i19

export PYTHONPATH=$TIMEPIX_I19:$PYTHONPATH

alias histogram='python3 ${TIMEPIX_I19}/timepix/time_histogram.py'
alias make_images='python3 ${TIMEPIX_I19}/timepix/timepix2M.py'
alias time_histogram=histogram
