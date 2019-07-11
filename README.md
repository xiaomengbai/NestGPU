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

Query JA:

SELECT S.s
FROM S
WHERE S.a IN ( SELECT MAX(R.a)
               FROM R
			   WHERE S.c = R.c);
			
#The new files : code_gen.py and ytree.py

the entry of the compiler is at the line 2149 in code_gen.py, gpudb_code_gen()
then will go to the line 2163, tree_node = ystree.ysmart_tree_gen(argv[1],argv[2]) if running the instr. like "python translate.py test/ssb_test/qex.sql test/ssb_test/ssb.schema"
and jump to the ystree.py at line 5085. 
In the ytree.py, I add the SubQNode class at line 1121, which is a modification of TwoJoinNode class, and the YSubExp class at line 526.
I also modify the class of FirstStepWhereCondition at line 2193 to enable subquery parsing, which is at line 2341 ~ 2345.
The YSubExp class is first used at line 101. The SubQNode is used at line 1600.