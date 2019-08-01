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

The debuging path is : code_gen.py 2163, ysmart_tree_gen() -> ytree.py 5096, build_plan_tree_from_a_select_node() -> 2839 convert_a_select_tree() -> 2588 FirstStepWhereCondition() -> 2370 __init__()

There are three functions in the init(), mainly changes are in utility_convert_where_condition_to_exp().

In utility_convert_where_condition_to_exp(), changing at line 2341.

After the processing of convert_a_select_tree(), back to line 2843 convert_to_initial_query_plan_tree() then go to the utility_convert_to_initial_plan_tree() at line 1661

The subquery is handled at line 1596 and finally return a SubQNode to t0 at line 2843.

The convert_to_binary_join_tree() should require more modification, but I keep returning the SubQNode itself now. Here we finish the build_plan_tree_from_a_select_node() and will go back to code_gen.py at line 2163

After that line, the functions generate_schema_file(), generate_loader() and generate_soa() may need no change, I guess.

I working on the generate_code() at line 2202 now.