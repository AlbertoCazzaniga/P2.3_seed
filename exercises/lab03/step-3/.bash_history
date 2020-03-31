ls
cmake .
make 
ls
./step-3 
$(spack location -i mpich)/bin/mpirun -np 4 ./step-3
$(spack location -i mpich)/bin/mpirun -np 12 ./step-3
$(spack location -i mpich)/bin/mpirun -np 2 ./step-3
exit
