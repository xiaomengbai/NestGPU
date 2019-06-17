# subquery
#compile with the following instructions
nvcc -O3 -arch=sm_60 subquerynested.cu -w -rdc=true -lcudadevrt -o sj
#to run it
./sj
#the function called
main -> preparation -> subquery_proc
the code before the "GPU-DB unnest time" is the orginal processing on GPU-DB
after that are my codes.
the "nestScanDynamicUnlimited_deviceFunc" is the entry
and the "recallScanUmlimited" is the nested or recursive GPU kernel.