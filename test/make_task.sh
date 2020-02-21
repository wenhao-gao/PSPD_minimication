#!/bin/bash

methods=('vanilla' 'bfgs' 'conjugate_gradient' 'adam'\
	 'vanilla_armijo' 'bfgs_armijo' 'conjugate_gradient_armijo'\
	 'vanilla_brent' 'bfgs_brent' 'conjugate_gradient_brent')

for method in ${methods[@]}
do
cat > task_time_${method}.sh << EOF
python ${method}_time.py -i time_complexity/helix_10.pdb
python ${method}_time.py -i time_complexity/helix_30.pdb
python ${method}_time.py -i time_complexity/helix_70.pdb
python ${method}_time.py -i time_complexity/helix_110.pdb
python ${method}_time.py -i time_complexity/helix_150.pdb
python ${method}_time.py -i time_complexity/helix_200.pdb
EOF
done
