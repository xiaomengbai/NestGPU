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