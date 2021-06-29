#! /bin/bash
#$ -pe smp 8
#$ -m ea
#$ -M fabio.federici@ukaea.uk

# The top four lines set the:
#     1) linux enviroment
#     2) number of threads to assign to the job (important if you parralise a job like in the python jobs below, this will
#        keep you out of trouble I got in when I was overloading the freia CPUs, lol)
#     3) tells the job manager to notify you of job completion/termination
#     4) email address to send notification to (I recomend setting up a sub-inbox just for job notification, that is what I did)
#
#
# The job will start in whatever directory you enviroment starts you in. This is likely your home dir /home/ffederici/
#
# changing my enviroment
source /home/ffederic/.bashrc
# going to the directory that I want to excecut my jobs from (useful if your file references
# only have relative pathdir)
cd /home/ffederic/work/analysis_scripts/scripts
# first python script (all the extra stuff is because I use argparse to pass info, variables and files)
python /home/ffederic/work/analysis_scripts/scripts/MASTU_pulse_process.py
# python /home/ffederic/work/analysis_scripts/operations/test20.py
# second python script
# python optimise_posterior.py --sample --polish '/home/dgahle/thesis/balmer_analysis/exp_general/baysar_output/solps_analysis/baysar_fit_wensing_pex_d2_ramp_TCV' --sample --posterior_file '/home/dgahle/thesis/balmer_analysis/exp_general/baysar_posteriors/solps_analysis/posterior_wensing_pex_d2_ramp_TCV_150143_chord_16' --save '/home/dgahle/thesis/balmer_analysis/exp_general/baysar_output/solps_analysis/baysar_fit_wensing_pex_d2_ramp_TCV_150143_chord_16' --num_threads 4
