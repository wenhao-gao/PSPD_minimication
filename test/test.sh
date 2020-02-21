#!/bin/bash


name=('sa' 'sass' 'saa')
line=(1 2 3)
for i in ${line[@]}
do
for n in ${name[@]}
do
cat > test_${i}_${n}.py << EOF
sa ${i}
sas ${n}
sa
EOF
done
done
