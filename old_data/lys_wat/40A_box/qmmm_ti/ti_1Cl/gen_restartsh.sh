perl -i -ne 'if(/total_steps/) {s/1/2/;} print' RESTART
perl -ne 's/nvt\.xml/RESTART/; s/>/>>/; print' run.sh > restart.sh
