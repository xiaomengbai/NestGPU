# subquery
#Compile with the following instructions
nvcc -O3 -arch=sm_60 umlimitedNested.cu -w -rdc=true -lcudadevrt -o un
#To run it
./un
#The function called
main -> preparation -> subquery_proc

the code before the "GPU-DB unnest time" (around line 671)is the orginal processing on GPU-DB, after that are my codes.

the "nestScanDynamicUnlimited_deviceFunc" is the entry

and the "recallScanUmlimited" is the nested or recursive GPU kernel.

the "nestScanIterative_deviceFunc" is the iterative kernel, supportting Query JA. 183 seconds on 4 million outer table and 1 million inner table.