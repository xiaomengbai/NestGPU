# subquery
#compile with the following instructions
nvcc -O3 -arch=sm_60 subquerynested.cu -w -rdc=true -lcudadevrt -o sj
