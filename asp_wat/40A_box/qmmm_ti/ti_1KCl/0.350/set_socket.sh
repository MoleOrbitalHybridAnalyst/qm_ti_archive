perl -i -ne 's/XXX/'$1'/; print' driver.py
perl -i -ne 's/XXX/'$1'/; print' nvt.xml
perl -i -ne 's/LMP/'$2'/; print' lmp.in
perl -i -ne 's/LMP/'$2'/; print' nvt.xml
