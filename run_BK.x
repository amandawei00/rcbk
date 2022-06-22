qsub -cwd -V -N BK -l h_data=4G,h_rt=8:00:00,highp -pe shared 5 $PWD/run.bash
