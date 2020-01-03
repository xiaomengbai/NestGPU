#! /usr/bin/python
"""
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import sys
import commands
import os
import copy
import ystree
import correlation
import config
import pickle

schema = None
keepInGpu = 1
baseIndent = " " * 4
optimization = None

merge = lambda l1, l2: [ (l1[i], l2[i]) for i in range(0, min(len(l1), len(l2))) ]
unmerge = lambda l: ( [ l[i][0] for i in range(0, len(l)) ], [ l[i][1] for i in range(0, len(l)) ] )
preload_threshold = 1024 * 1024 * 1024
loaded_table_list = {}
cols_to_index = []

"""
Get the value of the configurable variables from config.py
"""

joinType = config.joinType
POS = config.POS
SOA = config.SOA
CODETYPE = config.CODETYPE

PID = config.PID
DTYPE = config.DTYPE

class columnAttr(object):
    type = None
    size = None

    def __init__ (self):
        self.type = ""
        self.size = 0

class JoinTranslation(object):
    dimTables = None
    factTables = None
    joinNode = None
    dimIndex = None
    factIndex = None
    outIndex = None
    outAttr = None
    outPos = None

    def __init__ (self):
        self.dimTables = []
        self.factTables = []
        self.joinNode = []
        self.dimIndex = []
        self.factIndex = []
        self.outIndex = []
        self.outAttr = []
        self.outPos = []

    def __repr__(self):
        print "dimTables: ", self.dimTables
        print "factTables: ", self.factTables
        print "joinNode: ", self.joinNode
        print "dimIndex: ", self.dimIndex
        print "factIndex: ", self.factIndex
        print "outIndex: ", self.outIndex
        print "outAttr: ", self.outAttr
        print "outPos: ", self.outPos

def __get_gb_exp__(exp,tmp_list):
    if not isinstance(exp,ystree.YFuncExp):
        return

    if exp.func_name in ["SUM","AVG","COUNT","MAX","MIN"]:
        tmp_list.append(exp)
    else:
        for x in exp.parameter_list:
            __get_gb_exp__(x,tmp_list)


def get_gbexp_list(exp_list,gb_exp_list):
    for exp in exp_list:
        if not isinstance(exp,ystree.YFuncExp):
            continue
        tmp_list = []
        __get_gb_exp__(exp,tmp_list)
        for tmp in tmp_list:
            tmp_bool = False
            for gb_exp in gb_exp_list:
                if tmp.compare(gb_exp) is True:
                    tmp_bool = True
                    break
            if tmp_bool is False:
                gb_exp_list.append(tmp)

"""
get_tables() gets the translation information for join, agg and order by nodes.
Currently we only support star schema queries.
We assume that the dimTable is always the right child of the join node.
"""

def get_tables(tree, joinAttr, aggNode, orderbyNode):

    # The leaf is the fact table
    if isinstance(tree, ystree.TableNode):
        joinAttr.factTables.append(tree)
        return

    # copy into orderbyNode
    elif isinstance(tree, ystree.OrderByNode):
        obNode = copy.deepcopy(tree)
        orderbyNode.append(obNode)
        get_tables(tree.child, joinAttr, aggNode, orderbyNode)

    # copy into aggNode
    elif isinstance(tree, ystree.GroupByNode):

        gbNode = copy.deepcopy(tree)
        aggNode.append(gbNode)
        get_tables(tree.child, joinAttr, aggNode, orderbyNode)

    elif isinstance(tree, ystree.TwoJoinNode):

        leftIndex = []
        rightIndex = []
        leftAttr = []
        rightAttr = []
        leftPos = []
        rightPos = []

        # copy into joinAttr
        newNode = copy.deepcopy(tree)
        joinAttr.joinNode.insert(0,newNode)

        for exp in tree.select_list.tmp_exp_list:

            index = tree.select_list.tmp_exp_list.index(exp)
            if isinstance(exp,ystree.YRawColExp):
                colAttr = columnAttr()
                colAttr.type = exp.column_type
                if exp.table_name == "LEFT":

                    if joinType == 0:
                        leftIndex.append(exp.column_name)

                    elif joinType == 1:
                        newExp = ystree.__trace_to_leaf__(tree,exp,False) # Get the real table name
                        leftIndex.append(newExp.column_name)

                    leftAttr.append(colAttr)
                    leftPos.append(index)

                elif exp.table_name == "RIGHT":

                    if joinType == 0:
                        rightIndex.append(exp.column_name)

                    elif joinType == 1:
                        newExp = ystree.__trace_to_leaf__(tree,exp,False)
                        rightIndex.append(newExp.column_name)

                    rightAttr.append(colAttr)
                    rightPos.append(index)

        # index: column id in the original table
        outList= []
        outList.append(leftIndex)
        outList.append(rightIndex)

        # attr: column name in the original table
        outAttr = []
        outAttr.append(leftAttr)
        outAttr.append(rightAttr)

        # pos: column id in the join table
        outPos = []
        outPos.append(leftPos)
        outPos.append(rightPos)

        joinAttr.outIndex.insert(0,outList)
        joinAttr.outAttr.insert(0, outAttr)
        joinAttr.outPos.insert(0, outPos)

        pkList = tree.get_pk()
        # def exp_list_to_str(exp_list):
        #     return map(lambda exp: exp.evaluate(), exp_list)
        # print "pkList[0]: ", exp_list_to_str(pkList[0])
        # print "pkList[1]: ", exp_list_to_str(pkList[1])

        if (len(pkList[0]) != len(pkList[1])):
            print 1/0

        if joinType == 0:
            for exp in pkList[0]:
                colIndex = 0
                if isinstance(tree.left_child, ystree.TableNode):
                    colIndex = -1
                    for tmp in tree.left_child.select_list.tmp_exp_list:
                        if exp.column_name == tmp.column_name:
                            colIndex = tree.left_child.select_list.tmp_exp_list.index(tmp)
                            break
                    if colIndex == -1:
                        print 1/0
                else:
                    colIndex = exp.column_name

        elif joinType == 1:
            for exp in pkList[0]:
                newExp = ystree.__trace_to_leaf__(tree,exp,True)
                colIndex = newExp.column_name

        joinAttr.factIndex.insert(0, colIndex)

        for exp in pkList[1]:
            colIndex = 0
            if isinstance(tree.right_child, ystree.TableNode):
                colIndex = -1
                for tmp in tree.right_child.select_list.tmp_exp_list:
                    if exp.column_name == tmp.column_name:
                        colIndex = tree.right_child.select_list.tmp_exp_list.index(tmp)
                        break
                if colIndex == -1:
                    print 1/0
            else:
                colIndex = exp.column_name
            joinAttr.dimIndex.insert(0, colIndex)

        if isinstance(tree.right_child, ystree.TableNode):
            joinAttr.dimTables.insert(0, tree.right_child)

        get_tables(tree.left_child, joinAttr, aggNode, orderbyNode)


"""
Translate the type defined in the schema into the supported
type in the translated c program.
Currently only three types are supported:
    INT, FLOAT and STRING.
"""

def to_ctype(colType):

    if colType in ["INTEGER","DATE"]:
        return "INT";
    elif colType in ["TEXT"]:
        return "STRING"
    elif colType in ["DECIMAL"]:
        return "FLOAT"

"""
Get the length of a given column.
"""

def type_length(tn, colIndex, colType):
    if colType in ["INTEGER", "DATE"]:
        return "sizeof(int)"
    elif colType in ["TEXT"]:
        colLen = schema[tn].column_list[colIndex].column_others
        return str(colLen)
    elif colType in ["DECIMAL"]:
        return "sizeof(float)"

"""
Get the exp information from the where expresion.
"""

def get_where_attr(exp, whereList, relList, conList):
    if isinstance(exp, ystree.YFuncExp):
        if exp.func_name in ["AND", "OR"]:
            for x in exp.parameter_list:
                if isinstance(x, ystree.YFuncExp):
                    get_where_attr(x,whereList, relList, conList)
                elif isinstance(x, ystree.YRawColExp):
                    whereList.append(x)
        else:
            relList.append(exp.func_name)
            for x in exp.parameter_list:
                if isinstance(x, ystree.YRawColExp):
                    whereList.append(x)
                elif isinstance(x, ystree.YConsExp):
                    if x.ref_col != None:
                        conList.append(x.ref_col)
                    else:
                        conList.append(x.cons_value)
                elif isinstance(x, ystree.YFuncExp) and (x.func_name == "SUBQ" or x.func_name == "LIST"):
                    conList.append(x)

    elif isinstance(exp, ystree.YRawColExp):
        whereList.append(exp)


"""
Generate a new where list where no duplate columns exist.
Return the number of columns in the new list.
"""

def count_whereList(wlist, tlist):

    for col in wlist:
        colExist = False
        for x in tlist:
            if x.compare(col) is True:
                colExist = True
                break
        if colExist is False:
            tlist.append(col)

    return len(tlist)

"""
count the nested level of the where condition.
"""

def count_whereNested(exp):
    count = 0

    if isinstance(exp, ystree.YFuncExp):
        if exp.func_name in ["AND", "OR"]:
            for x in exp.parameter_list:
                max = 0
                if isinstance(x,ystree.YFuncExp) and x.func_name in ["AND","OR"]:
                    max +=1
                    max += count_whereNest(x)
                    if max > count:
                        count = max

    return count

class mathExp:
    opName = None
    leftOp = None
    rightOp = None
    value = None

    def __init__(self):
        self.opName = None
        self.leftOp = None
        self.rightOp = None
        self.value = None

    def addOp(self, exp):

        if isinstance(exp,ystree.YRawColExp):
            self.opName = "COLUMN_" + exp.column_type
            self.value = exp.column_name
        elif isinstance(exp,ystree.YConsExp):
            self.opName = "CONS"
            self.value = exp.cons_value
        elif isinstance(exp,ystree.YFuncExp):
            self.opName = exp.func_name
            leftExp = exp.parameter_list[0]
            rightExp = exp.parameter_list[1]

            self.leftOp = mathExp()
            self.rightOp = mathExp()
            self.leftOp.addOp(leftExp)
            self.rightOp.addOp(rightExp)

### print the mathExp in c

def printMathFunc(fo,prefix, mathFunc):

    if mathFunc.opName == "COLUMN":
        print >>fo, prefix + ".op = NOOP;"
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = 0;"
        print >>fo, prefix + ".opType = COLUMN;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    elif mathFunc.opName == "COLUMN_DECIMAL":
        print >>fo, prefix + ".op = NOOP;"
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = 0;"
        print >>fo, prefix + ".opType = COLUMN_DECIMAL;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    elif mathFunc.opName == "COLUMN_INTEGER":
        print >>fo, prefix + ".op = NOOP;"
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = 0;"
        print >>fo, prefix + ".opType = COLUMN_INTEGER;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    elif mathFunc.opName == "CONS":
        print >>fo, prefix + ".op = NOOP;"
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = 0;"
        print >>fo, prefix + ".opType = CONS;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    else:
        print >>fo, prefix + ".op = " + mathFunc.opName + ";"
        print >>fo, prefix + ".opNum = 2;"
        print >>fo, prefix + ".exp = (long) malloc(sizeof(struct mathExp) * 2);"
        prefix1 = "((struct mathExp *)" + prefix + ".exp)[0]"
        prefix2 = "((struct mathExp *)"+ prefix + ".exp)[1]"
        printMathFunc(fo,prefix1,mathFunc.leftOp)
        printMathFunc(fo,prefix2,mathFunc.rightOp)

"""
generate_col_list gets all the columns that will be scannned for a given table node.
@indexList stores the index of each column.
@colList stores the columnExp for each column.
"""

def generate_col_list(tn, indexList, colList):

    for col in tn.select_list.tmp_exp_list:
        if col.column_name not in indexList:
            indexList.append(col.column_name)
            colList.append(col)

    if tn.where_condition is not None:
        whereList = []
        relList = []
        conList = []
        get_where_attr(tn.where_condition.where_condition_exp,whereList,relList,conList)
        for col in whereList:
            if col.column_name not in indexList:
                indexList.append(col.column_name)
                colList.append(col)
        for con in conList:
            if isinstance(con, ystree.YFuncExp) and con.func_name == "SUBQ":
                for par in con.parameter_list:
                    if isinstance(par, ystree.YRawColExp) and par.column_name not in indexList:
                        indexList.append(par.column_name)
                        colList.append(par)

"""
generate_code generates CUDA/OpenCL codes from the query plan tree.
Currently we only generate CUDA/OpenCL codes for star schema queries.

Several configurable variables (in config.py):
    @CODETYPE determines whether CUDA or OpenCL codes should be generated.
    0 represents CUDA and 1 represents OpenCL.

    @joinType determines whether we should generate invisible joins for star
    schema queries. 0 represents normal join and 1 represents invisible join.

    @POS describes where the data are stored in the host memory and how the
    codes should be generated. 0 means data are stored in pageable host
    memory and data are explicitly transferred. 1 means data are stored in
    pinned host memory and data are explicitly transferred. 2 means data are
    stored in pinned host memory and the kernel will directly access the data
    without explicit data transferring. 3 means data are stored in disk and only
    mapped to host memory.
"""

def generate_code(tree):

    global baseIndent
    """
    First check whether the value of each configurable variable is valid.
    All should be integers.
    """

    if CODETYPE not in [0,1]:
        print "Error! The value of CODETYPE can only be 0 or 1."
        exit(-1)

    if POS not in [0,1,2,3]:
        print "Error! The value of POS can only be 0,1,2,3."
        exit(-1)

    if joinType not in [0,1]:
        print "Error! The value of JOINTYPE can only be 0 or 1."
        exit(-1)

    DTYPE_STR = ""
    if CODETYPE == 1:
        if PID not in [0,1,2,3]:
            print "Error for PID!"
            exit(-1)

        if DTYPE not in [0,1,2]:
            print "Error! The value of DTYPE can only be 0,1,2."
            exit(-1)

        if DTYPE == 0:
            DTYPE_STR = "CL_DEVICE_TYPE_GPU"
        elif DTYPE == 1:
            DTYPE_STR = "CL_DEVICE_TYPE_CPU"
        elif DTYPE == 2:
            DTYPE_STR = "CL_DEVICE_TYPE_ACCELERATOR"

    if CODETYPE==0:
        fo = open("driver.cu","w")
    else:
        fo = open("driver.cpp","w")

    print >>fo, "/* This file is generated by code_gen.py */"
    print >>fo, "#include <stdio.h>"
    print >>fo, "#include <stdlib.h>"
    print >>fo, "#include <sys/types.h>"
    print >>fo, "#include <sys/stat.h>"
    print >>fo, "#include <fcntl.h>"
    print >>fo, "#include <sys/mman.h>"
    print >>fo, "#include <string.h>"
    print >>fo, "#include <unistd.h>"
    print >>fo, "#include <malloc.h>"
    print >>fo, "#include <time.h>"
    print >>fo, "#include <getopt.h>"
    print >>fo, "#include <linux/limits.h>"
    print >>fo, "#include \"../include/common.h\""

    if joinType == 0:
        print >>fo, "#include \"../include/hashJoin.h\""
    else:
        print >>fo, "#include \"../include/inviJoin.h\""

    print >>fo, "#include \"../include/schema.h\""
    print >>fo, "#include \"../include/mempool.h\""

    if CODETYPE == 0:
        print >>fo, "#include \"../include/cpuCudaLib.h\""
        print >>fo, "#include \"../include/gpuCudaLib.h\""
        #print >>fo, "extern struct tableNode* tableScan(struct scanNode *,struct statistic *);"
        print >>fo, "extern struct tableNode* tableScan(struct scanNode *,struct statistic *, bool, bool, bool, int *, int *, int *);"
        #Indexing functions
        print >>fo, "extern void createIndex (struct tableNode *, int, int, struct statistic *);"
        if joinType == 0:
            #print >>fo, "extern struct tableNode* hashJoin(struct joinNode *, struct statistic *);"
            print >>fo, "extern struct tableNode* hashJoin(struct joinNode *, struct statistic *, bool, bool, bool);"
        else:
            print >>fo, "extern struct tableNode* inviJoin(struct joinNode *, struct statistic *);"
        print >>fo, "extern struct tableNode* groupBy(struct groupByNode *,struct statistic *, bool, bool);"
        print >>fo, "extern struct tableNode* orderBy(struct orderByNode *, struct statistic *);"
        print >>fo, "extern char* materializeCol(struct materializeNode * mn, struct statistic *);"

    else:
        print >>fo, "#include <CL/cl.h>"
        print >>fo, "#include <string>"
        print >>fo, "#include \"../include/gpuOpenclLib.h\"\n"
        print >>fo, "#include\"../include/cpuOpenclLib.h\""
        print >>fo, "using namespace std;"
        print >>fo, "extern const char * createProgram(string, int *);"

        print >>fo, "extern struct tableNode* tableScan(struct scanNode *,struct clContext *, struct statistic *);"
        if joinType == 0:
            print >>fo, "extern struct tableNode* hashJoin(struct joinNode *, struct clContext *, struct statistic *);"
        else:
            print >>fo, "extern struct tableNode* inviJoin(struct joinNode *, struct clContext *, struct statistic *);"
        print >>fo, "extern struct tableNode* groupBy(struct groupByNode *, struct clContext *, struct statistic *);"
        print >>fo, "extern struct tableNode* orderBy(struct orderByNode *, struct clContext *, struct statistic *);"
        print >>fo, "extern char * materializeCol(struct materializeNode * mn, struct clContext *, struct statistic *);"

    if "--idx" in optimization:
        print >>fo, "#include <thrust/execution_policy.h>"
        print >>fo, "#include <thrust/system/cuda/execution_policy.h>"
        print >>fo, "#include <thrust/binary_search.h>"

    indent = baseIndent
    print >>fo, "\n#define CHECK_POINTER(p) do {\\"
    print >>fo, indent + "if(p == NULL){   \\"
    print >>fo, indent + baseIndent + "perror(\"Failed to allocate host memory\");    \\"
    print >>fo, indent + baseIndent + "exit(-1);      \\"
    print >>fo, indent + "}} while(0)"

    print >>fo, "\nint main(int argc, char ** argv){\n"

    if CODETYPE == 1:
        print >>fo, indent + "int psc = 0;"
        print >>fo, indent + "void * clTmp;"
        print >>fo, indent + "const char * ps = createProgram(\"kernel.cl\",&psc);"
        print >>fo, indent + "struct clContext context;"
        print >>fo, indent + "cl_uint numP;"
        print >>fo, indent + "cl_int error = 0;"
        print >>fo, indent + "cl_device_id device;"
        print >>fo, indent + "clGetPlatformIDs(0,NULL,&numP);"
        print >>fo, indent + "cl_platform_id * pid = new cl_platform_id[numP];"
        print >>fo, indent + "clGetPlatformIDs(numP, pid, NULL);"
        print >>fo, indent + "clGetDeviceIDs(pid[" + str(PID) + "]," + DTYPE_STR +", 1, &device, NULL);"
        print >>fo, indent + "context.context = clCreateContext(0, 1, &device, NULL, NULL, &error);"
        print >>fo, indent + "cl_command_queue_properties prop = 0;"
        print >>fo, indent + "prop |= CL_QUEUE_PROFILING_ENABLE;"
        print >>fo, indent + "context.queue = clCreateCommandQueue(context.context, device, prop, &error);"
        print >>fo, indent + "context.program = clCreateProgramWithSource(context.context, psc, (const char **)&ps, 0, &error);"
        print >>fo, indent + "error = clBuildProgram(context.program, 0, 0 , \"-I .\" , 0, 0);\n"

    else:
        print >>fo, indent + "/* For initializing CUDA device */"
        print >>fo, indent + "int * cudaTmp;"
        print >>fo, indent + "cudaMalloc((void**)&cudaTmp,sizeof(int));"
        print >>fo, indent + "cudaFree(cudaTmp);\n"


    print >>fo, indent + "int table;"
    print >>fo, indent + "int long_index;"
    print >>fo, indent + "char path[PATH_MAX];"
    print >>fo, indent + "int setPath = 0;"
    print >>fo, indent + "struct option long_options[] = {"
    print >>fo, indent + baseIndent + "{\"datadir\",required_argument,0,'0'}"
    print >>fo, indent + "};\n"

    print >>fo, indent + "while((table=getopt_long(argc,argv,\"\",long_options,&long_index))!=-1){"
    print >>fo, indent + baseIndent + "switch(table){"
    print >>fo, indent + baseIndent * 2 + "case '0':"
    print >>fo, indent + baseIndent * 3 + "setPath = 1;"
    print >>fo, indent + baseIndent * 3 + "strcpy(path,optarg);"
    print >>fo, indent + baseIndent * 3 + "break;"
    print >>fo, indent + baseIndent + "}"
    print >>fo, indent + "}\n"

    print >>fo, indent + "if(setPath == 1)"
    print >>fo, indent + baseIndent + "chdir(path);\n"

    print >>fo, indent + "struct timespec start, end;"
    print >>fo, indent + "struct timespec diskStart, diskEnd;"
    print >>fo, indent + "double diskTotal = 0;"

    print >>fo, indent + "clock_gettime(CLOCK_REALTIME,&start);"
    print >>fo, indent + "struct statistic pp;"
    print >>fo, indent + "pp.total = pp.kernel = pp.pcie = 0;\n"

    #For tableScan profiling
    print >>fo, indent + "pp.buildIndexTotal = 0;"
    print >>fo, indent + "pp.tableScanTotal = 0;"
    print >>fo, indent + "pp.tableScanCount = 0;"
    print >>fo, indent + "pp.whereMemCopy_s1 = 0;"
    print >>fo, indent + "pp.dataMemCopy_s2 = 0;"
    print >>fo, indent + "pp.scanTotal_s3 = 0;"
    print >>fo, indent + "pp.preScanTotal_s4 = 0;"
    print >>fo, indent + "pp.preScanCount_s4 = 0;"
    print >>fo, indent + "pp.preScanResultMemCopy_s5 = 0;"
    print >>fo, indent + "pp.dataMemCopyOther_s6 = 0;"
    print >>fo, indent + "pp.materializeResult_s7 = 0;"
    print >>fo, indent + "pp.finalResultMemCopy_s8 = 0;"
    print >>fo, indent + "pp.create_tableNode_S01 = 0;"
    print >>fo, indent + "pp.mallocRes_S02 = 0;"
    print >>fo, indent + "pp.deallocateBuffs_S03 = 0;"
    print >>fo, indent + "pp.getIndexPos_idxS1 = 0;"
    print >>fo, indent + "pp.getRange_idxS2 = 0;"
    print >>fo, indent + "pp.convertMemToElement_idxS3 = 0;"
    print >>fo, indent + "pp.getMapping_idxS4 = 0;"
    print >>fo, indent + "pp.setBitmapZeros_idxS5 = 0;"
    print >>fo, indent + "pp.buildBitmap_idxS6 = 0;"
    print >>fo, indent + "pp.countScanKernel_countS1 = 0;"
    print >>fo, indent + "pp.scanImpl_countS2 = 0;" 
    print >>fo, indent + "pp.preallocBlockSums_scanImpl_S1 = 0;"
    print >>fo, indent + "pp.prescanArray_scanImpl_S2 = 0;"
    print >>fo, indent + "pp.deallocBlockSums_scanImpl_S3 = 0;"
    print >>fo, indent + "pp.setVar_prescan_S1 = 0;"
    print >>fo, indent + "pp.preScanKernel_prescan_S2 = 0;"
    print >>fo, indent + "pp.uniformAddKernel_prescan_S3 = 0;\n"

    #For join profiling
    print >>fo, indent + "pp.join_totalTime = 0;"
    print >>fo, indent + "pp.join_callTimes = 0;"
    print >>fo, indent + "pp.joinProf_step1_allocateMem = 0;"
    print >>fo, indent + "pp.joinProf_step2_buildHash = 0;"
    print >>fo, indent + "pp.joinProf_step21_allocateMem = 0;"
    print >>fo, indent + "pp.joinProf_step22_Count_hash_num = 0;"
    print >>fo, indent + "pp.joinProf_step23_scanImpl = 0;"
    print >>fo, indent + "pp.joinProf_step24_buildhash_kernel_memcopy = 0;"
    print >>fo, indent + "pp.joinProf_step3_join = 0;"
    print >>fo, indent + "pp.joinProf_step31_allocateMem = 0;"
    print >>fo, indent + "pp.joinProf_step32_exclusiveScan = 0;"
    print >>fo, indent + "pp.joinProf_step33_prob = 0;"
    print >>fo, indent + "pp.joinProf_step4_deallocate = 0;\n"
    
    #Create mem pool
    print >>fo, indent + "init_mempool();";
    print >>fo, indent + "init_gpu_mempool(&gpu_inner_mp);";
    print >>fo, indent + "init_gpu_mempool(&gpu_inter_mp);\n";

    generate_code_for_loading_tables(fo, indent, tree)

    generate_code_for_a_tree(fo, tree, 0)

    print >>fo, "\n";
    print >>fo, indent + "destroy_mempool();";
    print >>fo, indent + "destroy_gpu_mempool(&gpu_inner_mp);";
    print >>fo, indent + "destroy_gpu_mempool(&gpu_inter_mp);\n";

    print >>fo, indent + "clock_gettime(CLOCK_REALTIME, &end);"
    print >>fo, indent + "double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;"
    print >>fo, indent + "printf(\"<--Disk Load Time-->           : %lf\\n\", diskTotal/(1000*1000));"
    #print >>fo, indent + "printf(\"PCIe Time: %lf\\n\",pp.pcie);"
    #print >>fo, indent + "printf(\"Kernel Time: %lf\\n\",pp.kernel);"

    print >>fo, indent + "printf(\"\\n\");"
    print >>fo, indent + "printf(\"<--Build index time-->         : %lf\\n\", pp.buildIndexTotal/(1000*1000));\n"
    print >>fo, indent + "printf(\"\\n\");"

    print >>fo, indent + "printf(\"<---TableScan()--->\\n\");"
    print >>fo, indent + "printf(\"Total time      : %lf\\n\", pp.tableScanTotal/(1000*1000));"
    print >>fo, indent + "printf(\"Calls           : %d\\n\", pp.tableScanCount);\n"
    print >>fo, indent + "printf(\"Step 1 - memCopy where clause                 : %lf\\n\", pp.whereMemCopy_s1/(1000*1000));"
    print >>fo, indent + "printf(\"Step 2 - memCopy predicate col                : %lf\\n\", pp.dataMemCopy_s2/(1000*1000));"
    print >>fo, indent + "printf(\"Step 3 - Scan                                 : %lf\\n\", pp.scanTotal_s3/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.1 - Get index position             : %lf\\n\", pp.getIndexPos_idxS1/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.2 - Get range                      : %lf\\n\", pp.getRange_idxS2/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.3 - Convert addrs to elements      : %lf\\n\", pp.convertMemToElement_idxS3/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.4 - Get mapping position           : %lf\\n\", pp.getMapping_idxS4/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.5 - Set bitmap to zero             : %lf\\n\", pp.setBitmapZeros_idxS5/(1000*1000));"
    print >>fo, indent + "printf(\"Idx Step 3.6 - Build bitmap                   : %lf\\n\", pp.buildBitmap_idxS6/(1000*1000));"
    print >>fo, indent + "printf(\"Step 4 - CountRes(PreScan)                    : %lf\\n\", pp.preScanTotal_s4/(1000*1000));"
    print >>fo, indent + "printf(\"PreScan Step 4.1 - Count selected rows kernel : %lf\\n\", pp.countScanKernel_countS1/(1000*1000));"
    print >>fo, indent + "printf(\"PreScan Step 4.2 - scanImpl time              : %lf\\n\", pp.scanImpl_countS2/(1000*1000));"
    print >>fo, indent + "printf(\"scanImpl Step 4.2.1 - preallocBlockSums time  : %lf\\n\", pp.preallocBlockSums_scanImpl_S1/(1000*1000));"
    print >>fo, indent + "printf(\"scanImpl Step 4.2.2 - prescanArray time       : %lf\\n\", pp.prescanArray_scanImpl_S2/(1000*1000));"
    print >>fo, indent + "printf(\"scanImpl Step 4.2.3 - deallocBlockSums time   : %lf\\n\", pp.deallocBlockSums_scanImpl_S3/(1000*1000));"
    print >>fo, indent + "printf(\"prescan Step 4.2.3.1 - set variables time     : %lf\\n\", pp.setVar_prescan_S1/(1000*1000));"
    print >>fo, indent + "printf(\"prescan Step 4.2.3.2 - prescan Kernel time    : %lf\\n\", pp.preScanKernel_prescan_S2/(1000*1000));"
    print >>fo, indent + "printf(\"prescan Step 4.2.3.3 - uniformAdd Kernel time : %lf\\n\", pp.uniformAddKernel_prescan_S3/(1000*1000));"
    print >>fo, indent + "printf(\"Step 5 - memReturn countRes                   : %lf\\n\", pp.preScanResultMemCopy_s5/(1000*1000));"
    print >>fo, indent + "printf(\"Step 6 - Copy rest of columns                 : %lf\\n\", pp.dataMemCopyOther_s6/(1000*1000));"
    print >>fo, indent + "printf(\"Step 7 - Materialize result                   : %lf\\n\", pp.materializeResult_s7/(1000*1000));"
    print >>fo, indent + "printf(\"Step 8 - Copy final result                    : %lf\\n\", pp.finalResultMemCopy_s8/(1000*1000));"
    print >>fo, indent + "printf(\"Other 1 - Create tableNode                    : %lf\\n\", pp.create_tableNode_S01/(1000*1000));"
    print >>fo, indent + "printf(\"Other 2 - Malloc res                          : %lf\\n\", pp.mallocRes_S02/(1000*1000));"
    print >>fo, indent + "printf(\"Other 3 - Deallocate buffers                  : %lf\\n\", pp.deallocateBuffs_S03/(1000*1000));"
    print >>fo, indent + "printf(\"<----------------->\");"
    print >>fo, indent + "printf(\"\\n\");"

    print >>fo, indent + "printf(\"<---HashJoin()--->\\n\");"
    print >>fo, indent + "printf(\"Total time      : %lf\\n\", pp.join_totalTime/(1000*1000));"
    print >>fo, indent + "printf(\"Calls           : %d\\n\", pp.join_callTimes);\n"
    print >>fo, indent + "printf(\"Step 1 - Allocate memory for intermediate results : %lf\\n\", pp.joinProf_step1_allocateMem/(1000*1000));"
    print >>fo, indent + "printf(\"Step 2 - Build hashTable                          : %lf\\n\", pp.joinProf_step2_buildHash/(1000*1000));"
    print >>fo, indent + "printf(\"Step 2.1 - Allocate memory                        : %lf\\n\", pp.joinProf_step21_allocateMem/(1000*1000));"    
    print >>fo, indent + "printf(\"Step 2.2 - Count_hash_num                         : %lf\\n\", pp.joinProf_step22_Count_hash_num/(1000*1000));"    
    print >>fo, indent + "printf(\"Step 2.3 - scanImpl                               : %lf\\n\", pp.joinProf_step23_scanImpl/(1000*1000));"    
    print >>fo, indent + "printf(\"Step 2.4 - build_hash_table (+ memCopy op)        : %lf\\n\", pp.joinProf_step24_buildhash_kernel_memcopy/(1000*1000));"    
    print >>fo, indent + "printf(\"Step 3 - Join                                     : %lf\\n\", pp.joinProf_step3_join/(1000*1000));"
    print >>fo, indent + "printf(\"Step 3.1 - Allocate memory                        : %lf\\n\", pp.joinProf_step31_allocateMem/(1000*1000));"
    print >>fo, indent + "printf(\"Step 3.2 - Exclusive scan                         : %lf\\n\", pp.joinProf_step32_exclusiveScan/(1000*1000));"
    print >>fo, indent + "printf(\"Step 3.3 - Prob and memcpy ops                    : %lf\\n\", pp.joinProf_step33_prob/(1000*1000));" 
    print >>fo, indent + "printf(\"Step 4 - De-allocate memory                       : %lf\\n\", pp.joinProf_step4_deallocate/(1000*1000));"
    print >>fo, indent + "printf(\"<----------------->\");"
    print >>fo, indent + "printf(\"\\n\");"

    print >>fo, indent + "printf(\"Total Time: %lf\\n\", timeE/(1000*1000));"
    print >>fo, "}\n"

    fo.close()

def get_subqueries(tree, subq_list):

    if isinstance(tree, ystree.OrderByNode) or isinstance(tree, ystree.GroupByNode):
        get_subqueries(tree.child, subq_list)

    elif isinstance(tree, ystree.TwoJoinNode):
        get_subqueries(tree.left_child, subq_list)
        get_subqueries(tree.right_child, subq_list)

    elif isinstance(tree, ystree.TableNode) and tree.where_condition != None:
        where_exp = tree.where_condition.where_condition_exp
        def __get_subqueries__(exp, subq_list):
            if exp is None or not isinstance(exp, ystree.YFuncExp):
                return
            if exp.func_name != "SUBQ":
                map(lambda par: __get_subqueries__(par, subq_list), exp.parameter_list)
            else:
                subq_list.append(exp)

        __get_subqueries__(where_exp, subq_list)
        return

def generate_code_for_a_tree(fo, tree, lvl):

    global baseIndent
    indent = (lvl * 3 + 1) * baseIndent

    var_subqRes = "subqRes" + str(lvl)

    print ">>> Generate code for a tree <<<"
    tree.debug(0)

    resultNode = "result"

    print >>fo, indent + "struct tableNode *" + resultNode + ";"
    print >>fo, indent + "char * " + var_subqRes + ";\n"

    tree_result = generate_code_for_a_node(fo, indent, lvl, tree)

    print >>fo, indent + resultNode + " = " + tree_result + ";"

    print >>fo, indent + "struct materializeNode mn;"
    print >>fo, indent + "mn.table = "+resultNode + ";"
    if CODETYPE == 0:
        print >>fo, indent + "char *final = materializeCol(&mn, &pp);"

        if lvl == 0:
            print >>fo, indent + "printMaterializedTable(mn, final);"

    else:
        print >>fo, indent + "materializeCol(&mn, &context,&pp);"
        print >>fo, indent + "freeTable(" + resultNode + ");\n"

    if CODETYPE == 1:
        print >>fo, indent + "clReleaseCommandQueue(context.queue);"
        print >>fo, indent + "clReleaseContext(context.context);"
        print >>fo, indent + "clReleaseProgram(context.program);\n"

def generate_code_for_a_subquery(fo, lvl, rel, con, tupleNum, tableName, indexDict, currentNode, passInPos):

    indent = (lvl * 3 + 2) * baseIndent
    var_subqRes = "subqRes" + str(lvl)

    subq_id = con.parameter_list[0].cons_value
    pass_in_cols = con.parameter_list[1:]
    sub_tree = ystree.subqueries[subq_id]

    subq_select_list = sub_tree.select_list.tmp_exp_list
    if len(subq_select_list) > 1:
        print "ERROR: more than one column are selected in a subquery!"
        sub_tree.debug(0)
        print 1/0

    selectItem = subq_select_list[0]
    subq_res_size = 0
    if isinstance(selectItem, ystree.YFuncExp):
        # Assume the aggregation function with only one parameter, i.e., the column
        subq_res_col = selectItem.parameter_list[0]
    elif isinstance(selectItem, ystree.YRawColExp):
        subq_res_col = selectItem
    else:
        print "ERROR: Unknown select column type in a subquery: ", selectItem.evaluate()
        print 1/0

    if rel in ["EQ", "LTH", "GTH", "LEQ", "GEQ", "NOT_EQ"]:
        constant_len_res = True
        subq_res_size = "sizeof(float)"
    else:
        constant_len_res = False
        subq_res_size = type_length(subq_res_col.table_name, subq_res_col.column_name, subq_res_col.column_type)


    print >>fo, ""
    print >>fo, indent + "// Process the subquery"

    if isinstance(currentNode, ystree.TwoJoinNode):
        pass_in_cols = map(lambda c: ystree.__trace_to_leaf__(currentNode, c, False), pass_in_cols)

    for col in pass_in_cols:
        colLen = type_length(col.table_name, col.column_name, col.column_type)
        pass_in_var = "_" + col.table_name + "_" + str(col.column_name)
        # print >>fo, indent + "char *" + pass_in_var + " = (char *)malloc(" + colLen + ");"
        # print >>fo, indent + "CHECK_POINTER(" + pass_in_var + ");"
        print >>fo, indent + "char *" + pass_in_var + " = (char *)alloc_mempool(" + colLen + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"


    print >>fo, indent + var_subqRes + " = (char *)malloc(" + (subq_res_size if constant_len_res else "sizeof(char *)") + " * " + tupleNum + ");"
    print >>fo, indent + "CHECK_POINTER(" + var_subqRes + ");\n"
    # print >>fo, indent + var_subqRes + " = (char *)alloc_mempool(" + (subq_res_size if constant_len_res else "sizeof(char *)") + " * " + tupleNum + ");"
    # print >>fo, indent + "MEMPOOL_CHECK();\n"

    # traverse all query plan trees to find tables and their columns
    columns_to_gpu = {}
    __traverse_tree(sub_tree, generate_col_lists(columns_to_gpu))
    for key, value in columns_to_gpu.items():
        # should consider if a column is not pre-loaded
        fullIndexList, fullColList = unmerge(loaded_table_list[key])
        t_name = key.lower() + "Table"
        for pair in value:
            print >>fo, indent + "transferTableColumnToGPU(" + t_name + ", " + str(fullIndexList.index(pair[0])) + ");"


    # Begin semi-join correlated columns in a sub-query
    def __extract_correlated_pairs(node, pair_list):
        if node.where_condition is not None and node.where_condition.where_condition_exp is not None:
            where_exp = node.where_condition.where_condition_exp
            is_passed_in_par = lambda par: isinstance(par, ystree.YConsExp) and par.ref_col is not None
            def __extract_correlated_pairs_exp(exp, _pair_list):
                if isinstance(exp, ystree.YFuncExp) and len(exp.parameter_list) == 2 and any(map(is_passed_in_par, exp.parameter_list)):
                    if is_passed_in_par(exp.parameter_list[0]):
                        if isinstance(node, ystree.TableNode):
                            pair_list.append((exp.parameter_list[0], exp.parameter_list[1]))
                        # may consider TwoJoinNode and trace to a leaf node for the real table name
                        # elif isinstance(node, ystree.TwoJoinNode)
                    else:
                        if isinstance(node, ystree.TableNode):
                            pair_list.append((exp.parameter_list[1], exp.parameter_list[0]))

            def extract_correlated_pairs_exp(_pair_list):
                return lambda e: __extract_correlated_pairs_exp(e, _pair_list)

            ystree.__traverse_exp_map__(where_exp, extract_correlated_pairs_exp(pair_list))

    def extract_correlated_pairs(pair_list):
        return lambda n: __extract_correlated_pairs(n, pair_list)

    correlated_pairs = []
    if "--idx" in optimization:
        __traverse_tree(sub_tree, extract_correlated_pairs(correlated_pairs))

    if len(correlated_pairs) > 0:
        print >>fo, ""
        print >>fo, indent + "char *passed_in_col, *correlated_col;"
        print >>fo, indent + "char *correlated_col_sort;\n"

    for pair in correlated_pairs:
        if isinstance(pair[1], ystree.YRawColExp):
            # check if a column has been pre-loaded
            is_col_preloaded = lambda col: loaded_table_list[col.table_name] is not None and any(map(lambda c: col.compare(c[1]), loaded_table_list[col.table_name]))
            col_idx_preloaded = lambda col: map(lambda c: col.compare(c[1]), loaded_table_list[col.table_name]).index(True)
            if is_col_preloaded(pair[0].ref_col) and is_col_preloaded(pair[1]):
                passed_in_idx = indexDict[pair[0].ref_col.table_name + "." + str(pair[0].ref_col.column_name)]
                correlated_table_node = pair[1].table_name.lower() + "Table"
                correlated_idx = col_idx_preloaded(pair[1])
                # may consider different data type
                index_type = "(int *)"
                print >>fo, indent + "// Semi-join ", pair[0].ref_col.table_name + "." + str(pair[0].ref_col.column_name), " and ", pair[1].table_name + "." + str(pair[1].column_name)
                print >>fo, indent + "if(" + tableName + "->dataPos[" + str(passed_in_idx) + "] == MEM)"
                print >>fo, indent + baseIndent + "transferTableColumnToGPU(" + tableName + ", " + str(passed_in_idx) + ");"
                print >>fo, indent + "passed_in_col = " + "(char *)(" + tableName + "->content[" +  str(passed_in_idx) + "]);"
                print >>fo, indent + "correlated_col = " + correlated_table_node + "->content[" + str(correlated_idx) + "];\n"

                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&correlated_col_sort, " + correlated_table_node + "->attrTotalSize[" + str(correlated_idx) + "]) );\n"

                print >>fo, indent + "if(" + correlated_table_node + "->dataPos[" + str(correlated_idx) + "] == MEM)"
                print >>fo, indent + baseIndent + "CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(correlated_col_sort, correlated_col, " + correlated_table_node + "->attrTotalSize[" + str(correlated_idx) + "], cudaMemcpyHostToDevice) );"
                print >>fo, indent + "else if(" + correlated_table_node + "->dataPos[" + str(correlated_idx) + "] == GPU)"
                print >>fo, indent + baseIndent + "CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(correlated_col_sort, correlated_col, " + correlated_table_node + "->attrTotalSize[" + str(correlated_idx) + "], cudaMemcpyDeviceToDevice) );\n"

                correlated_indices = pair[1].table_name.lower() + str(pair[1].column_name) + "_idx"
                print >>fo, indent + "int *" + correlated_indices + ";"
                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&" + correlated_indices + ", " + correlated_table_node + "->tupleNum * sizeof(int)) );"
                print >>fo, indent + "assign_index<<<2048, 256>>>(" + correlated_indices + ", " + correlated_table_node + "->tupleNum);"
                print >>fo, indent + "thrust::sort_by_key(thrust::device, " + index_type + "correlated_col_sort, " + index_type + "correlated_col_sort + " + correlated_table_node + "->tupleNum, " + correlated_indices + ");\n"

                passed_in_lo = pair[0].ref_col.table_name.lower() + str(pair[0].ref_col.column_name) + "_lo"
                passed_in_hi = pair[0].ref_col.table_name.lower() + str(pair[0].ref_col.column_name) + "_hi"
                print >>fo, indent + "int *" + passed_in_lo + ", *" + passed_in_hi + ";"
                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&" + passed_in_lo + ", " + tableName + "->tupleNum * sizeof(int)) );"
                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&" + passed_in_hi + ", " + tableName + "->tupleNum * sizeof(int)) );\n"

                print >>fo, indent + "thrust::lower_bound(thrust::device, " + index_type + "correlated_col_sort, " + index_type + "correlated_col_sort + " + correlated_table_node + "->tupleNum, " + index_type + "passed_in_col, " + index_type + "passed_in_col + " + tableName + "->tupleNum, " + passed_in_lo + ");"
                print >>fo, indent + "thrust::upper_bound(thrust::device, " + index_type + "correlated_col_sort, " + index_type + "correlated_col_sort + " + correlated_table_node + "->tupleNum, " + index_type + "passed_in_col, " + index_type + "passed_in_col + " + tableName + "->tupleNum, " + passed_in_hi + ");\n"

                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaFree(correlated_col_sort) );\n"

    print >>fo, indent + "char *free_pos = freepos_mempool();"
    print >>fo, indent + "char *free_gpu_pos = freepos_gpu_mempool(&gpu_inter_mp);";
    print >>fo, indent + "for(int tupleid = 0; tupleid < " + tupleNum + "; tupleid++){"

    for col in pass_in_cols:
        pass_in_var = "_" + col.table_name + "_" + str(col.column_name)
        colLen = type_length(col.table_name, col.column_name, col.column_type)
        if passInPos == "MEM":
            print >>fo, indent + baseIndent + "memcpy(" + pass_in_var + ", (char *)(" + tableName + "->content[" +  str(indexDict[col.table_name + "." + str(col.column_name)]) + "]) + tupleid * " + colLen + ", " + colLen + ");"
        elif passInPos == "GPU":
            print >>fo, indent + baseIndent + "CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(" + pass_in_var + ", (char *)(" + tableName + "->content[" + str(indexDict[col.table_name + "." + str(col.column_name)]) + "]) + tupleid * " + colLen + ", " + colLen + ", cudaMemcpyDeviceToHost) );"
        else:
            print "ERROR: Unknown pass in value position: ", passInPos
            exit(99)

    print >>fo, indent + baseIndent + "{"

    generate_code_for_a_tree(fo, sub_tree, lvl + 1)

    print >>fo, indent + baseIndent * 2 + ""
    if constant_len_res:
        print >>fo, indent + baseIndent * 2 + "mempcpy(" + var_subqRes + " + tupleid * " + subq_res_size + ", final, " + subq_res_size + ");"
    else:
        print >>fo, indent + baseIndent * 2 + "((char **)" + var_subqRes + ")[tupleid] = (char *)malloc(sizeof(int) + " + subq_res_size + " * mn.table->tupleNum);"
        print >>fo, indent + baseIndent * 2 + "CHECK_POINTER( ((char **)" + var_subqRes + ")[tupleid] );"
        print >>fo, indent + baseIndent * 2 + "*(int *)(((char **)" + var_subqRes + ")[tupleid]) = mn.table->tupleNum;"
        print >>fo, indent + baseIndent * 2 + "mempcpy(((char **)" + var_subqRes + ")[tupleid] + sizeof(int), final, " + subq_res_size + " * mn.table->tupleNum);"
    print >>fo, indent + baseIndent * 2 + "freeto_mempool(free_pos);"
    print >>fo, indent + baseIndent * 2 + "freeto_gpu_mempool(&gpu_inter_mp, free_gpu_pos);"
    print >>fo, indent + baseIndent + "}"
    print >>fo, indent + "}"

    for col in pass_in_cols:
        pass_in_var = "_" + col.table_name + "_" + str(col.column_name)
        #print >>fo, indent + "free(" + pass_in_var + ");"

    for pair in correlated_pairs:
        if isinstance(pair[1], ystree.YRawColExp):
            correlated_indices = pair[1].table_name.lower() + str(pair[1].column_name) + "_idx"
            print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaFree(" + correlated_indices + ") );"
            passed_in_lo = pair[0].ref_col.table_name.lower() + str(pair[0].ref_col.column_name) + "_lo"
            passed_in_hi = pair[0].ref_col.table_name.lower() + str(pair[0].ref_col.column_name) + "_hi"
            print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaFree(" + passed_in_lo + ") );"
            print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC( cudaFree(" + passed_in_hi + ") );"


    print >>fo, ""

"""
gpudb_code_gen: entry point for code generation.
"""

def gpudb_code_gen(argv):
    pwd = os.getcwd()
    resultDir = "./src"
    utilityDir = "./utility"

    if CODETYPE == 0:
        codeDir = "./cuda"
    else:
        codeDir = "./opencl"

    # includeDir = "./include"
    schemaFile = None

    #Exec query
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        tree_node = ystree.ysmart_tree_gen(argv[1],argv[2])
        if tree_node is None:
            exit(-1)

    #Data loading
    elif len(sys.argv) == 2:
        schemaFile = ystree.ysmart_get_schema(argv[1])

    global schema
    schema = ystree.global_table_dict

    if os.path.exists(resultDir) is False:
        os.makedirs(resultDir)

    os.chdir(resultDir)
    if os.path.exists(codeDir) is False:
        os.makedirs(codeDir)

    os.chdir(pwd)
    os.chdir(resultDir)
    os.chdir(codeDir)

    if len(sys.argv) >= 3:
        if len(sys.argv) > 3:
            global optimization
            optimization = argv[3]
        generate_code(tree_node)

    os.chdir(pwd)

def generate_code_for_a_node(fo, indent, lvl, node):

    if isinstance(node, ystree.TwoJoinNode):
        return generate_code_for_a_two_join_node(fo, indent, lvl, node)
    elif isinstance(node, ystree.GroupByNode):
        return generate_code_for_a_group_by_node(fo, indent, lvl, node)
    elif isinstance(node, ystree.OrderByNode):
        return generate_code_for_a_order_by_node(fo, indent, lvl, node)
    elif isinstance(node, ystree.TableNode):
        return generate_code_for_a_table_node(fo, indent, lvl, node)
    elif isinstance(node, ystree.SelectProjectNode):
        return generate_code_for_a_select_project_node(fo, indent, lvl, node)

def generate_code_for_a_select_project_node(fo, indent, lvl, spn):

    inputNode = generate_code_for_a_node(fo, indent, lvl, spn.child)

    return inputNode

def generate_code_for_a_order_by_node(fo, indent, lvl, obn):

    inputNode = generate_code_for_a_node(fo, indent, lvl, obn.child)

    orderby_exp_list = obn.order_by_clause.orderby_exp_list
    odLen = len(orderby_exp_list)

    resultNode = inputNode + "_ob"
    print >>fo, indent + "struct tableNode * " + resultNode + ";"

    print >>fo, indent + "{\n"
    indent += baseIndent

    # print >>fo, indent + "struct orderByNode * odNode = (struct orderByNode *) malloc(sizeof(struct orderByNode));"
    # print >>fo, indent + "CHECK_POINTER(odNode);"
    print >>fo, indent + "struct orderByNode * odNode = (struct orderByNode *)alloc_mempool(sizeof(struct orderByNode));"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + "odNode->table = " + inputNode +";"
    print >>fo, indent + "odNode->orderByNum = " + str(odLen) + ";"
    # print >>fo, indent + "odNode->orderBySeq = (int *) malloc(sizeof(int) * odNode->orderByNum);"
    # print >>fo, indent + "CHECK_POINTER(odNode->orderBySeq);"
    # print >>fo, indent + "odNode->orderByIndex = (int *) malloc(sizeof(int) * odNode->orderByNum);"
    # print >>fo, indent + "CHECK_POINTER(odNode->orderByIndex);"
    print >>fo, indent + "odNode->orderBySeq = (int *)alloc_mempool(sizeof(int) * odNode->orderByNum);"
    print >>fo, indent + "odNode->orderByIndex = (int *)alloc_mempool(sizeof(int) * odNode->orderByNum);"
    print >>fo, indent + "MEMPOOL_CHECK();"


    for i in range(0, odLen):
        seq = obn.order_by_clause.order_indicator_list[i]
        if seq == "ASC":
            print >>fo, indent + "odNode->orderBySeq[" + str(i) + "] = ASC;"
        else:
            print >>fo, indent + "odNode->orderBySeq[" + str(i) + "] = DESC;"

        print >>fo, indent + "odNode->orderByIndex[" + str(i) + "] = " + str(orderby_exp_list[i].column_name) + ";"

    if CODETYPE == 0:
        print >>fo, indent + resultNode + " = orderBy(odNode, &pp);"
    else:
        print >>fo, indent + resultNode + " = orderBy(odNode, &context, &pp);"

    # print >>fo, indent + "freeOrderByNode(odNode);\n"
    print >>fo, indent + "freeOrderByNode(odNode, false);\n"

    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}\n"

    return resultNode

def generate_code_for_a_group_by_node(fo, indent, lvl, gbn):

    inputNode = generate_code_for_a_node(fo, indent, lvl, gbn.child)

    gb_exp_list = gbn.group_by_clause.groupby_exp_list
    select_list = gbn.select_list.tmp_exp_list
    selectLen = len(select_list)
    gbLen = len(gb_exp_list)

    resultNode = inputNode + "_gb"
    print >>fo, indent + "struct tableNode * " + resultNode + ";"

    print >>fo, indent + "{\n"
    indent += baseIndent

    # print >>fo, indent + "struct groupByNode * gbNode = (struct groupByNode *) malloc(sizeof(struct groupByNode));"
    # print >>fo, indent + "CHECK_POINTER(gbNode);"
    print >>fo, indent + "struct groupByNode * gbNode = (struct groupByNode *)alloc_mempool(sizeof(struct groupByNode));"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + "gbNode->table = " + inputNode +";"
    print >>fo, indent + "gbNode->groupByColNum = " + str(gbLen) + ";"
    # print >>fo, indent + "gbNode->groupByIndex = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"
    # print >>fo, indent + "CHECK_POINTER(gbNode->groupByIndex);"
    # print >>fo, indent + "gbNode->groupByType = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"
    # print >>fo, indent + "CHECK_POINTER(gbNode->groupByType);"
    # print >>fo, indent + "gbNode->groupBySize = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"
    # print >>fo, indent + "CHECK_POINTER(gbNode->groupBySize);"
    print >>fo, indent + "gbNode->groupByIndex = (int *)alloc_mempool(sizeof(int) * " + str(gbLen) + ");"
    print >>fo, indent + "gbNode->groupByType = (int *)alloc_mempool(sizeof(int) * " + str(gbLen) + ");"
    print >>fo, indent + "gbNode->groupBySize = (int *)alloc_mempool(sizeof(int) * " + str(gbLen) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();"


    for i in range(0,gbLen):
        exp = gb_exp_list[i]
        if isinstance(exp, ystree.YRawColExp):
            print >>fo, indent + "gbNode->groupByIndex[" + str(i) + "] = " + str(exp.column_name) + ";"
            print >>fo, indent + "gbNode->groupByType[" + str(i) + "] = gbNode->table->attrType[" + str(exp.column_name) + "];"
            print >>fo, indent + "gbNode->groupBySize[" + str(i) + "] = gbNode->table->attrSize[" + str(exp.column_name) + "];"
        elif isinstance(exp, ystree.YConsExp):
            print >>fo, indent + "gbNode->groupByIndex[" + str(i) + "] = -1;"
            print >>fo, indent + "gbNode->groupByType[" + str(i) + "] = INT;"
            print >>fo, indent + "gbNode->groupBySize[" + str(i) + "] = sizeof(int);"
        else:
            print 1/0


    print >>fo, indent + "gbNode->outputAttrNum = " + str(selectLen) + ";"
    # print >>fo, indent + "gbNode->attrType = (int *) malloc(sizeof(int) *" + str(selectLen) + ");"
    # print >>fo, indent + "CHECK_POINTER(gbNode->attrType);"
    # print >>fo, indent + "gbNode->attrSize = (int *) malloc(sizeof(int) *" + str(selectLen) + ");"
    # print >>fo, indent + "CHECK_POINTER(gbNode->attrSize);"
    # print >>fo, indent + "gbNode->tupleSize = 0;"
    # print >>fo, indent + "gbNode->gbExp = (struct groupByExp *) malloc(sizeof(struct groupByExp) * " + str(selectLen) + ");"
    print >>fo, indent + "gbNode->attrType = (int *)alloc_mempool(sizeof(int) *" + str(selectLen) + ");"
    print >>fo, indent + "gbNode->attrSize = (int *)alloc_mempool(sizeof(int) *" + str(selectLen) + ");"
    print >>fo, indent + "gbNode->tupleSize = 0;"
    print >>fo, indent + "gbNode->gbExp = (struct groupByExp *)alloc_mempool(sizeof(struct groupByExp) * " + str(selectLen) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();"



    for i in range(0,selectLen):
        exp = select_list[i]
        if isinstance(exp, ystree.YFuncExp):

            print >>fo, indent + "gbNode->tupleSize += sizeof(float);"
            print >>fo, indent + "gbNode->attrType[" + str(i) + "] = FLOAT;"
            print >>fo, indent + "gbNode->attrSize[" + str(i) + "] = sizeof(float);"
            print >>fo, indent + "gbNode->gbExp["+str(i)+"].func = " + exp.func_name + ";"
            para = exp.parameter_list[0]
            mathFunc = mathExp()
            mathFunc.addOp(para)
            prefix = indent + "gbNode->gbExp[" + str(i) + "].exp"
            printMathFunc(fo,prefix, mathFunc)

        elif isinstance(exp, ystree.YRawColExp):
            colIndex = exp.column_name
            print >>fo, indent + "gbNode->attrType[" + str(i) + "] = " + inputNode + "->attrType[" + str(colIndex) + "];"
            print >>fo, indent + "gbNode->attrSize[" + str(i) + "] = " + inputNode + "->attrSize[" + str(colIndex) + "];"
            print >>fo, indent + "gbNode->tupleSize += "+inputNode + "->attrSize[" + str(colIndex) + "];"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].func = NOOP;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.op = NOOP;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.exp = NULL;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opNum = 1;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opType = COLUMN;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opValue = " + str(exp.column_name) + ";"

        else:
            if exp.cons_type == "INTEGER":
                print >>fo, indent + "gbNode->attrType[" + str(i) + "] = INT;"
                print >>fo, indent + "gbNode->attrSize[" + str(i) + "] = sizeof(int);"
                print >>fo, indent + "gbNode->tupleSize += sizeof(int);"
            elif exp.cons_type == "FLOAT":
                print >>fo, indent + "gbNode->attrType[" + str(i) + "] = FLOAT;"
                print >>fo, indent + "gbNode->attrSize[" + str(i) + "] = sizeof(float);"
                print >>fo, indent + "gbNode->tupleSize += sizeof(float);"
            else:
                print 1/0

            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].func = NOOP;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.op = NOOP;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.exp = NULL;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opNum = 1;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opType = CONS;"
            print >>fo, indent + "gbNode->gbExp[" + str(i) + "].exp.opValue = " + str(exp.cons_value) + ";"

    if CODETYPE == 0:
        print >>fo, indent + "// space required on GPU for inner data structures of GroupBy()"
        print >>fo, indent + "size_t new_size_groupby = (gpu_inner_mp.free - gpu_inner_mp.base);"
        print >>fo, indent + "new_size_groupby += gbNode->table->totalAttr * sizeof(char *); //gpuContent"
        print >>fo, indent + "int gbConstant = (gbNode->groupByColNum == 1 && gbNode->groupByIndex[0] == -1);"
        print >>fo, indent + "if (gbConstant != 1) {"
        print >>fo, indent + baseIndent + "new_size_groupby += 3 * sizeof(int) * gbNode->groupByColNum; // gpuGbType + gpuGbSize + gpuGbIndex"
        print >>fo, indent + baseIndent + "new_size_groupby += sizeof(int) * gbNode->table->tupleNum; // gpuGbKey"
        print >>fo, indent + baseIndent + "new_size_groupby += 3 * sizeof(int) * HSIZE; // gpu_hashNum + gpu_groupNum + gpu_psum"
        print >>fo, indent + baseIndent + "new_size_groupby += sizeof(int); // gpuGbCount"
        print >>fo, indent + "}"
        print >>fo, indent + "new_size_groupby += sizeof(char *) * gbNode->outputAttrNum; // gpuResult"
        print >>fo, indent + "new_size_groupby += 2 * sizeof(int) * gbNode->outputAttrNum; // gpuGbType + gpuGbSize"
        print >>fo, indent + "new_size_groupby += sizeof(struct groupByExp) * gbNode->outputAttrNum; // gpuGbExp"
        print >>fo, indent + "new_size_groupby += (sizeof(struct mathExp) * 2) * gbNode->outputAttrNum; // tmpMath"
        print >>fo, indent + "resize_gpu_mempool(&gpu_inner_mp, new_size_groupby);"
        print >>fo, indent + "char *origin_pos_groupby = freepos_gpu_mempool(&gpu_inner_mp);"
        print >>fo, indent + resultNode + " = groupBy(gbNode, &pp, true, true);"
        print >>fo, indent + "freeto_gpu_mempool(&gpu_inner_mp, origin_pos_groupby);\n"
    else:
        print >>fo, indent + resultNode + " = groupBy(gbNode, &context, &pp);"

    #print >>fo, indent + "freeGroupByNode(gbNode);\n"
    print >>fo, indent + "freeGroupByNode(gbNode, false);\n"


    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}\n"

    return resultNode

def generate_code_for_a_two_join_node(fo, indent, lvl, jn):

    leftName, rightName = generate_code_for_a_node(fo, indent, lvl, jn.left_child), generate_code_for_a_node(fo, indent, lvl, jn.right_child)
    var_subqRes = "subqRes" + str(lvl)

    if leftName is None or rightName is None:
        print 1/0

    print >>fo, indent + "// Join two tables: " + leftName + ", " + rightName # conditions?
    resName = leftName + "_" + rightName
    print >>fo, indent + "struct tableNode *" + resName + ";\n"

    print >>fo, indent + "{\n"

    jName = "jNode"
    indent += baseIndent
    print >>fo, indent + "struct joinNode " + jName + ";"
    print >>fo, indent + jName + ".leftTable = " + leftName + ";"
    print >>fo, indent + jName + ".rightTable = " + rightName + ";"

    lOutList = []
    rOutList = []

    lPosList = []
    rPosList = []

    lAttrList = []
    rAttrList = []

    selectList = jn.select_list.tmp_exp_list
    for exp in jn.select_list.tmp_exp_list:
        index = jn.select_list.tmp_exp_list.index(exp)
        if isinstance(exp, ystree.YRawColExp):
            colAttr = columnAttr()
            colAttr.type = exp.column_type
            if exp.table_name == "LEFT":
                if joinType == 0:
                    lOutList.append(exp.column_name)
                elif joinType == 1:
                    newExp = ystree.__trace_to_leaf__(jn, exp, False) # Get the real table name
                    lOutList.append(newExp.column_name)

                lAttrList.append(colAttr)
                lPosList.append(index)

            elif exp.table_name == "RIGHT":
                if joinType == 0:
                    rOutList.append(exp.column_name)
                elif joinType == 1:
                    newExp = ystree.__trace_to_leaf__(jn, exp, False)
                    rOutList.append(newExp.column_name)

                rAttrList.append(colAttr)
                rPosList.append(index)

    if jn.where_condition is not None:
        whereList = []
        relList = []
        conList = []
        get_where_attr(jn.where_condition.where_condition_exp, whereList, relList, conList)
        index = len(lPosList) + len(rPosList)
        for col in whereList:
            if (col.table_name == "LEFT" and col.column_name in lOutList) or (col.table_name == "RIGHT" and col.column_name in rOutList):
                continue

            colAttr = columnAttr()
            colAttr.type = col.column_type
            if col.table_name == "LEFT":
                if joinType == 0:
                    lOutList.append(col.column_name)
                elif joinType == 1:
                    newExp = ystree.__trace_to_leaf__(jn, col, False)
                    lOutList.append(newExp.column_name)
                lAttrList.append(colAttr)
                lPosList.append(index)
                index = index + 1
            elif col.table_name == "RIGHT":
                if joinType == 0:
                    rOutList.append(col.column_name)
                elif joinType == 1:
                    newExp = ystree.__trace_to_leaf__(jn, col, False)
                    rOutList.append(newExp.column_name)
                rAttrList.append(colAttr)
                rPosList.append(index)
                index = index + 1

        for con in conList:
            if isinstance(con, ystree.YFuncExp) and con.func_name == "SUBQ":
                for par in con.parameter_list:
                    if isinstance(par, ystree.YRawColExp):
                        if (par.table_name == "LEFT" and par.column_name in lOutList) or (par.table_name == "RIGHT" and par.column_name in rOutList):
                            continue

                        col = par
                        colAttr = columnAttr()
                        colAttr.type = col.column_type
                        if col.table_name == "LEFT":
                            if joinType == 0:
                                lOutList.append(col.column_name)
                            elif joinType == 1:
                                newExp = ystree.__trace_to_leaf__(jn, col, False)
                                lOutList.append(newExp.column_name)
                            lAttrList.append(colAttr)
                            lPosList.append(index)
                            index = index + 1
                        elif col.table_name == "RIGHT":
                            if joinType == 0:
                                rOutList.append(col.column_name)
                            elif joinType == 1:
                                newExp = ystree.__trace_to_leaf__(jn, col, False)
                                rOutList.append(newExp.column_name)
                            rAttrList.append(colAttr)
                            rPosList.append(index)
                            index = index + 1

    pkList = jn.get_pk()

    if (len(pkList[0]) != len(pkList[1])):
        print "ERROR: the join indices do not match!"
        exit(-1)

    leftIndex = None
    rightIndex = None
    if joinType == 0:
        for exp in pkList[0]:
            leftIndex = 0
            if isinstance(jn.left_child, ystree.TableNode):
                leftIndex = -1
                for tmp in jn.left_child.select_list.tmp_exp_list:
                    if exp.column_name == tmp.column_name:
                        leftIndex = jn.left_child.select_list.tmp_exp_list.index(tmp)
                        break
                if leftIndex == -1:
                    print 1/0
            else:
                leftIndex = exp.column_name

    elif joinType == 1:
        for exp in pkList[0]:
            newExp = ystree.__trace_to_leaf__(jn, exp, True)
            leftIndex = newExp.column_name


    for exp in pkList[1]:
        rightIndex = 0
        if isinstance(jn.right_child, ystree.TableNode):
            rightIndex = -1
            for tmp in jn.right_child.select_list.tmp_exp_list:
                if exp.column_name == tmp.column_name:
                    rightIndex = jn.right_child.select_list.tmp_exp_list.index(tmp)
                    break
            if rightIndex == -1:
                print 1/0
        else:
            rightIndex = exp.column_name

    if leftIndex is None or rightIndex is None:
        print "ERROR: Failed to find join indices for two tables"
        exit(-1)

    print >>fo, indent + jName + ".totalAttr = " + str(len(rOutList) + len(lOutList)) + ";"
    # print >>fo, indent + jName + ".keepInGpu = (int *) malloc(sizeof(int) * " + str(len(rOutList) + len(lOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".keepInGpu);"
    print >>fo, indent + jName + ".keepInGpu = (int *)alloc_mempool(sizeof(int) * " + str(len(rOutList) + len(lOutList)) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();"

    if keepInGpu == 0:
        print >>fo, indent + "for(int k=0; k<" + str(len(rOutList) + len(lOutList))  + "; k++)"
        print >>fo, indent + baseIndent + jName + ".keepInGpu[k] = 0;"
    else:
        print >>fo, indent + "for(int k=0; k<" + str(len(rOutList) + len(lOutList))  + "; k++)"
        print >>fo, indent + baseIndent + jName + ".keepInGpu[k] = 1;"

    print >>fo, indent + jName + ".leftOutputAttrNum = " + str(len(lOutList)) + ";"
    print >>fo, indent + jName + ".rightOutputAttrNum = " + str(len(rOutList)) + ";"

    # print >>fo, indent + jName + ".leftOutputAttrType = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".leftOutputAttrType);"
    # print >>fo, indent + jName + ".leftOutputIndex = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".leftOutputIndex);"
    # print >>fo, indent + jName + ".leftPos = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".leftPos);"
    print >>fo, indent + jName + ".leftOutputAttrType = (int *)alloc_mempool(sizeof(int)*" + str(len(lOutList)) + ");"
    print >>fo, indent + jName + ".leftOutputIndex = (int *)alloc_mempool(sizeof(int)*" + str(len(lOutList)) + ");"
    print >>fo, indent + jName + ".leftPos = (int *)alloc_mempool(sizeof(int)*" + str(len(lOutList)) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + jName + ".tupleSize = 0;"
    for j in range(0,len(lOutList)):
        ctype = to_ctype(lAttrList[j].type)
        print >>fo, indent + jName + ".leftOutputIndex[" + str(j) + "] = " + str(lOutList[j]) + ";"
        print >>fo, indent + jName + ".leftOutputAttrType[" + str(j) + "] = " + ctype + ";"
        print >>fo, indent + jName + ".leftPos[" + str(j) + "] = " + str(lPosList[j]) + ";"
        print >>fo, indent + jName + ".tupleSize += " + leftName + "->attrSize[" + str(lOutList[j]) + "];"

    # print >>fo, indent + jName + ".rightOutputAttrType = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".rightOutputAttrType);"
    # print >>fo, indent + jName + ".rightOutputIndex = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".rightOutputIndex);"
    # print >>fo, indent + jName + ".rightPos = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + jName + ".rightPos);"
    print >>fo, indent + jName + ".rightOutputAttrType = (int *)alloc_mempool(sizeof(int)*" + str(len(rOutList)) + ");"
    print >>fo, indent + jName + ".rightOutputIndex = (int *)alloc_mempool(sizeof(int)*" + str(len(rOutList)) + ");"
    print >>fo, indent + jName + ".rightPos = (int *)alloc_mempool(sizeof(int)*" + str(len(rOutList)) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();"
    for j in range(0,len(rOutList)):
        ctype = to_ctype(rAttrList[j].type)
        print >>fo, indent + jName + ".rightOutputIndex[" + str(j) + "] = " + str(rOutList[j]) + ";"
        print >>fo, indent + jName + ".rightOutputAttrType[" + str(j) + "] = " + ctype + ";"
        print >>fo, indent + jName + ".rightPos[" + str(j) + "] = " + str(rPosList[j]) + ";"
        print >>fo, indent + jName + ".tupleSize += " + rightName + "->attrSize[" + str(rOutList[j]) + "];"

    print >>fo, indent + jName + ".leftKeyIndex = " + str(leftIndex) + ";"
    print >>fo, indent + jName + ".rightKeyIndex = " + str(rightIndex) + ";"

    print >>fo, indent + "struct tableNode *joinRes;"
    if CODETYPE == 0:
        print >>fo, indent + "// space required on GPU for inner data structures of HashJoin()"
        print >>fo, indent + "int hsize = " + jName + ".rightTable->tupleNum; NP2(hsize);"
        print >>fo, indent + "size_t new_size_join = (gpu_inner_mp.free - gpu_inner_mp.base);"
        print >>fo, indent + "new_size_join += 3 * hsize * sizeof(int) + " + jName + ".rightTable->tupleNum * 2 * sizeof(int); // (gpu_hashNum + gpu_psum + gpu_psum1) + gpu_bucket"
        print >>fo, indent + "int threadNum = dim3(4096).x * dim3(256).x;"
        print >>fo, indent + "new_size_join += sizeof(int) * threadNum * 2; // gpu_count + gpu_resPsum"
        print >>fo, indent + "long filterSize = " + jName + ".leftTable->attrSize[" + jName + ".leftKeyIndex] * " + jName + ".leftTable->tupleNum;"
        print >>fo, indent + "new_size_join += filterSize * 3; // gpuFactFilter + filterNum + filterPsum"
        print >>fo, indent + "resize_gpu_mempool(&gpu_inner_mp, new_size_join);"
        print >>fo, indent + "char *origin_pos_join = freepos_gpu_mempool(&gpu_inner_mp);"
        print >>fo, indent + "joinRes = hashJoin(&" + jName + ", &pp, true, true, " + ("true" if lvl > 0 else "false") + ");"
        print >>fo, indent + "freeto_gpu_mempool(&gpu_inner_mp, origin_pos_join);\n"
    else:
        print >>fo, indent + "joinRes = hashJoin(&" + jName + ", &context, &pp);\n"

    if jn.where_condition is not None and not jn.where_condition.where_condition_exp.compare(jn.join_condition.where_condition_exp):
        expList = []
        def get_join_exps(exp, expList):
            if isinstance(exp, ystree.YFuncExp):
                if exp.func_name in ["AND", "OR"]:
                    for x in exp.parameter_list:
                        if isinstance(x, ystree.YFuncExp):
                            get_join_exps(x, expList)
                else:
                    expList.append(exp)

        get_join_exps(jn.where_condition.where_condition_exp, expList)

        relName = "joinRel"
        print >>fo, indent + "// Where conditions: " + jn.where_condition.where_condition_exp.evaluate()
        print >>fo, indent + "struct scanNode " + relName + ";"
        print >>fo, indent + relName + ".tn = joinRes;"
        print >>fo, indent + relName + ".hasWhere = 1;"
        print >>fo, indent + relName + ".whereAttrNum = " + str(len(expList)) + ";"
        # print >>fo, indent + relName + ".whereIndex = (int *)malloc(sizeof(int) * " + str(len(expList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".whereIndex);"
        print >>fo, indent + relName + ".whereIndex = (int *)alloc_mempool(sizeof(int) * " + str(len(expList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"
        print >>fo, indent + relName + ".outputNum = " + str(len(selectList)) + ";"
        # print >>fo, indent + relName + ".outputIndex = (int *)malloc(sizeof(int) * " + str(len(selectList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".outputIndex);"
        print >>fo, indent + relName + ".outputIndex = (int *)alloc_mempool(sizeof(int) * " + str(len(selectList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"


        for i in range(0, len(selectList)):
            print >>fo, indent + relName + ".outputIndex[" + str(i) + "] = " + str(i) + ";"


        def col2index_in_joinRes(col):
            if col.table_name == "LEFT":
                return lPosList[lOutList.index(col.column_name)]
            elif col.table_name == "RIGHT":
                return rPosList[rOutList.index(col.column_name)]

        where_col_dict = {}
        for i in range(0, len(expList)):
            exp = expList[i]
            if not isinstance(exp, ystree.YFuncExp) or len(exp.parameter_list) != 2:
                print "ERROR: doesn't support join condition: ", exp.evaluate()
                exit(99)

            left_col = exp.parameter_list[0]
            right_col = exp.parameter_list[1]
            if not isinstance(left_col, ystree.YRawColExp):
                print "ERROR: the left side of join must be a column!: ", left_col.evaluate()
                exit(99)

            if not isinstance(right_col, ystree.YRawColExp) and not (isinstance(right_col, ystree.YFuncExp) and right_col.func_name == "SUBQ"):
                print "ERROR: the right side of join must be a column or a subquery!: ", right_col.evaluate()
                exit(99)

            if left_col.evaluate() not in where_col_dict.keys():
                print >>fo, indent + relName + ".whereIndex[" + str(i) + "] = " + str(col2index_in_joinRes(left_col)) + ";"
                where_col_dict[left_col.evaluate()] = i

        if keepInGpu == 0:
            print >>fo, indent + relName + ".KeepInGpu = 0;"
        else:
            print >>fo, indent + relName + ".keepInGpu = 1;"

        # print >>fo, indent + relName + ".filter = (struct whereCondition *)malloc(sizeof(struct whereCondition));"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".filter);"
        print >>fo, indent + relName + ".filter = (struct whereCondition *)alloc_mempool(sizeof(struct whereCondition));"
        print >>fo, indent + "MEMPOOL_CHECK();"

        print >>fo, indent + "(" + relName + ".filter)->nested = 0;"
        print >>fo, indent + "(" + relName + ".filter)->expNum = " + str(len(expList)) + ";"
        # print >>fo, indent + "(" + relName + ".filter)->exp = (struct whereExp*)malloc(sizeof(struct whereExp) *" + str(len(expList)) + ");"
        # print >>fo, indent + "CHECK_POINTER((" + relName + ".filter)->exp);"
        print >>fo, indent + "(" + relName + ".filter)->exp = (struct whereExp*)alloc_mempool(sizeof(struct whereExp) *" + str(len(expList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"

        if jn.where_condition.where_condition_exp.func_name in ["AND","OR"]:
            print >>fo, indent + "(" + relName + ".filter)->andOr = " + jn.where_condition.where_condition_exp.func_name + ";"
        else:
            print >>fo, indent + "(" + relName + ".filter)->andOr = EXP;"

        for i in range(0, len(expList)):
            exp = expList[i]
            left_col = exp.parameter_list[0]
            right_col = exp.parameter_list[1]

            colType = left_col.column_type
            ctype = to_ctype(colType)

            print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].index    = " + str(where_col_dict[left_col.evaluate()]) + ";"

            print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].relation = " + exp.func_name + "_VEC;"

            if isinstance(right_col, ystree.YRawColExp):
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].dataPos  = GPU;"
                print >>fo, indent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &joinRes->content[" + str(col2index_in_joinRes(right_col)) + "], sizeof(void *));"
            else:
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].dataPos  = MEM;"

                indexDict = {}
                for j in range(0, len(lOutList)):
                    col = ystree.YRawColExp("LEFT", "")
                    col.column_name = lOutList[j]
                    indexDict[ystree.__trace_to_leaf__(jn, col, False).evaluate()] = lPosList[j]
                for j in range(0, len(rOutList)):
                    col = ystree.YRawColExp("RIGHT", "")
                    col.column_name = rOutList[j]
                    indexDict[ystree.__trace_to_leaf__(jn, col, False).evaluate()] = rPosList[j]

                generate_code_for_a_subquery(fo, lvl, exp.func_name, right_col, "joinRes->tupleNum", "joinRes", indexDict, jn, "GPU")

                print >>fo, indent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &" + var_subqRes + ", sizeof(void *));"

        if CODETYPE == 0:
            # print >>fo, indent + resName + " = tableScan(&" + relName + ", &pp);"
            print >>fo, indent + "// space required on GPU: gpuFilter + gpuPsum + gpuCount + gpuExp + padding"
            print >>fo, indent + "size_t new_size = (gpu_inner_mp.free - gpu_inner_mp.base) + joinRes->tupleNum * sizeof(int) + sizeof(int) * dim3(2048).x * dim3(256).x + sizeof(int) * dim3(2048).x * dim3(256).x + sizeof(struct whereExp) + 128;"
            print >>fo, indent + "resize_gpu_mempool(&gpu_inner_mp, new_size);"
            print >>fo, indent + "char *origin_pos = freepos_gpu_mempool(&gpu_inner_mp);"
            print >>fo, indent + resName + " = tableScan(&" + relName + ", &pp, true, true" + (", true" if lvl > 0 else ", false") + ", NULL, NULL, NULL);"
            print >>fo, indent + "freeto_gpu_mempool(&gpu_inner_mp, origin_pos);"
        else:
            print >>fo, indent + resName + " = tableScan(&" + relName + ", &context, &pp);"

        # print >>fo, indent + "freeScan(&" + relName + ");\n"
        print >>fo, indent + "freeScan(&" + relName + ", false);\n"

        if CODETYPE == 1:
            print >>fo, indent + "clFinish(context.queue);"

    else:
        print >>fo, indent + resName + " = joinRes;"

    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}\n"

    return resName

table_abbr = {
    "part" : "pa",
    "supplier" : "su",
    "customer" : "cu",
    "nation" : "na",
    "region" : "re",
    "partsupp" : "ps"
    }

table_refs = {
    "part" : 0,
    "supplier" : 0,
    "customer" : 0,
    "nation" : 0,
    "region" : 0,
    "partsupp" : 0
    }

def generate_code_for_a_table_node(fo, indent, lvl, tn):
    global loaded_table_list

    resName =  tn.table_name.lower()[0:2] if table_abbr[tn.table_name.lower()] == None else table_abbr[tn.table_name.lower()]
    resName = resName + str(table_refs[tn.table_name.lower()])
    table_refs[tn.table_name.lower()] = table_refs[tn.table_name.lower()] + 1
    tnName = tn.table_name.lower() + "Table"
    var_subqRes = "subqRes" + str(lvl)
    print >>fo, indent + "// Process the TableNode for " + tn.table_name.upper()
    print >>fo, indent + "struct tableNode *" + resName + ";"
    # print >>fo, indent + resName + " = (struct tableNode *)malloc(sizeof(struct tableNode));"
    # print >>fo, indent + "CHECK_POINTER("+ resName + ");"
    print >>fo, indent + resName + " = (struct tableNode *)alloc_mempool(sizeof(struct tableNode));"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + "initTable(" + resName + ");"

    print >>fo, indent + "{"
    indent += baseIndent

    indexList = []
    colList   = []
    generate_col_list(tn, indexList, colList)

    tablePreloaded = False
    selectList = tn.select_list.tmp_exp_list

    if tn.table_name in loaded_table_list.keys():
        tablePreloaded = True
        tnName = tn.table_name.lower() + "TablePartial"
        preloadName = tn.table_name.lower() + "Table"
        fullIndexList, fullColList = unmerge(loaded_table_list[tn.table_name])

        totalAttr = len(indexList)
        print >>fo, indent + "struct tableNode *" + tnName + ";"
        # print >>fo, indent + tnName + " = (struct tableNode *)malloc(sizeof(struct tableNode));"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + ");"
        # print >>fo, indent + tnName + "->totalAttr = " + str(totalAttr) + ";"
        # print >>fo, indent + tnName + "->attrType = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrType);"
        # print >>fo, indent + tnName + "->attrSize = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrSize);"
        # print >>fo, indent + tnName + "->attrIndex = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrIndex);"
        # print >>fo, indent + tnName + "->attrTotalSize = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrTotalSize);"
        # print >>fo, indent + tnName + "->dataPos = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->dataPos);"
        # print >>fo, indent + tnName + "->dataFormat = (int *) malloc(sizeof(int) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->dataFormat);"
        # print >>fo, indent + tnName + "->content = (char **)malloc(sizeof(char *) * " + str(totalAttr) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + tnName + "->content);\n"

        print >>fo, indent + tnName + " = (struct tableNode *)alloc_mempool(sizeof(struct tableNode));"
        print >>fo, indent + "MEMPOOL_CHECK();"
        print >>fo, indent + tnName + "->totalAttr = " + str(totalAttr) + ";"
        print >>fo, indent + tnName + "->attrType = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->attrSize = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->attrIndex = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->attrTotalSize = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->dataPos = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->dataFormat = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
        print >>fo, indent + tnName + "->content = (char **)alloc_mempool(sizeof(char *) * " + str(totalAttr) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();\n"


        print >>fo, indent + "int tuple_size = 0;"
        for i in range(0, totalAttr):

            fullIndexPos = lambda pidx: fullIndexList.index(indexList[pidx])
            print >>fo, indent + tnName + "->attrSize["        + str(i) + "] = " + preloadName + "->attrSize["        + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->attrIndex["       + str(i) + "] = " + preloadName + "->attrIndex["       + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->attrType["        + str(i) + "] = " + preloadName + "->attrType["        + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->dataPos["         + str(i) + "] = " + preloadName + "->dataPos["         + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->dataFormat["      + str(i) + "] = " + preloadName + "->dataFormat["      + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->attrTotalSize["   + str(i) + "] = " + preloadName + "->attrTotalSize["   + str(fullIndexPos(i)) + "];"
            print >>fo, indent + tnName + "->content["         + str(i) + "] = " + preloadName + "->content["         + str(fullIndexPos(i)) + "];"
            print >>fo, indent + "tuple_size += " + tnName + "->attrSize[" + str(i) + "];\n"

        print >>fo, indent + tnName + "->tupleSize = tuple_size;"
        print >>fo, indent + tnName + "->tupleNum = " + preloadName + "->tupleNum;\n"

        sortList = map( lambda c: int(c.split(".")[1]), filter( lambda c: c.split(".")[0] == tn.table_name, cols_to_index ) )
        sortListPart = list(set(indexList) & set(sortList))

        print >>fo, indent + tnName + "->colIdxNum = " + str(len(sortListPart)) + ";"
        print >>fo, indent + tnName + "->keepInGpuIdx = 1;"
        if len(sortListPart) > 0:
            # print >>fo, indent + tnName + "->colIdx = (int *)malloc(sizeof(int) * " + str(len(sortListPart)) + ");"
            # print >>fo, indent + "CHECK_POINTER(" + tnName + "->colIdx);"
            # print >>fo, indent + tnName + "->contentIdx = (int **)malloc(sizeof(int *) * " + str(len(sortListPart)) + ");"
            # print >>fo, indent + "CHECK_POINTER(" + tnName + "->contentIdx);"
            # print >>fo, indent + tnName + "->posIdx = (int **)malloc(sizeof(int *) * " + str(len(sortListPart)) + ");"
            # print >>fo, indent + "CHECK_POINTER(" + tnName + "->posIdx);\n"

            print >>fo, indent + tnName + "->colIdx = (int *)alloc_mempool(sizeof(int) * " + str(len(sortListPart)) + ");"
            print >>fo, indent + tnName + "->contentIdx = (int **)alloc_mempool(sizeof(int *) * " + str(len(sortListPart)) + ");"
            print >>fo, indent + tnName + "->posIdx = (int **)alloc_mempool(sizeof(int *) * " + str(len(sortListPart)) + ");"
            print >>fo, indent + "MEMPOOL_CHECK();\n"


            for i in range(0, len(sortListPart)):
                print >>fo, indent + tnName + "->colIdx[" + str(i) + "] = " + str( indexList.index(sortListPart[i]) ) + ";"
                print >>fo, indent + tnName + "->posIdx[" + str(i) + "] = " + preloadName + "->posIdx[" + str( sortList.index(sortListPart[i]) ) + "];"
                print >>fo, indent + tnName + "->contentIdx[" + str(i) + "] = " + preloadName + "->contentIdx[" + str( sortList.index(sortListPart[i]) ) + "];"

            #Also get index position
            print >>fo, indent + tnName + "->indexPos = "+ preloadName + "->indexPos;"
    else:
        tnName = generate_code_for_loading_a_table(fo, indent, tn.table_name, merge(indexList, colList))

    if tn.where_condition is not None:
        whereList = []
        relList = []
        conList = []

        get_where_attr(tn.where_condition.where_condition_exp, whereList, relList, conList)
        newWhereList = []
        whereLen = count_whereList(whereList, newWhereList)
        nested = count_whereNested(tn.where_condition.where_condition_exp)

        if nested != 0:
            print "Not supported yet: the where expression is too complicated"
            print 1/0

        relName = tn.table_name.lower() + "Rel"
        print >>fo, indent + "// Where conditions: " + tn.where_condition.where_condition_exp.evaluate()
        print >>fo, indent + "struct scanNode " + relName + ";"
        print >>fo, indent + relName + ".tn = " + tnName + ";"
        print >>fo, indent + relName + ".hasWhere = 1;"
        print >>fo, indent + relName + ".whereAttrNum = " + str(whereLen) + ";"
        # print >>fo, indent + relName + ".whereIndex = (int *)malloc(sizeof(int) * " + str(len(whereList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".whereIndex);"
        print >>fo, indent + relName + ".whereIndex = (int *)alloc_mempool(sizeof(int) * " + str(len(whereList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"
        print >>fo, indent + relName + ".outputNum = " + str(len(selectList)) + ";"
        # print >>fo, indent + relName + ".outputIndex = (int *)malloc(sizeof(int) * " + str(len(selectList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".outputIndex);"
        print >>fo, indent + relName + ".outputIndex = (int *)alloc_mempool(sizeof(int) * " + str(len(selectList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"

        for i in range(0, len(selectList)):
            colIndex = selectList[i].column_name
            outputIndex = indexList.index(colIndex)
            print >>fo, indent + relName + ".outputIndex[" + str(i) + "] = " + str(outputIndex) + ";"

        for i in range(0, len(newWhereList)):
            colIndex = indexList.index(newWhereList[i].column_name)
            print >>fo, indent + relName + ".whereIndex["+str(i) + "] = " + str(colIndex) + ";"

        if keepInGpu ==0:
            print >>fo, indent + relName + ".KeepInGpu = 0;"
        else:
            print >>fo, indent + relName + ".keepInGpu = 1;"

        # print >>fo, indent + relName + ".filter = (struct whereCondition *)malloc(sizeof(struct whereCondition));"
        # print >>fo, indent + "CHECK_POINTER(" + relName + ".filter);"
        print >>fo, indent + relName + ".filter = (struct whereCondition *)alloc_mempool(sizeof(struct whereCondition));"
        print >>fo, indent + "MEMPOOL_CHECK();"

        print >>fo, indent + "(" + relName + ".filter)->nested = 0;"
        print >>fo, indent + "(" + relName + ".filter)->expNum = " + str(len(whereList)) + ";"
        # print >>fo, indent + "(" + relName + ".filter)->exp = (struct whereExp*)malloc(sizeof(struct whereExp) *" + str(len(whereList)) + ");"
        # print >>fo, indent + "CHECK_POINTER((" + relName + ".filter)->exp);"
        print >>fo, indent + "(" + relName + ".filter)->exp = (struct whereExp*)alloc_mempool(sizeof(struct whereExp) *" + str(len(whereList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();"

        if tn.where_condition.where_condition_exp.func_name in ["AND","OR"]:
            print >>fo, indent + "(" + relName + ".filter)->andOr = " + tn.where_condition.where_condition_exp.func_name + ";"
        else:
            print >>fo, indent + "(" + relName + ".filter)->andOr = EXP;"

        indices = []
        los = []
        his = []
        for i in range(0, len(whereList)):
            colIndex = -1
            for j in range(0, len(newWhereList)):
                if newWhereList[j].compare(whereList[i]) is True:
                    colIndex = j
                    break

            if colIndex < 0:
                print 1/0

            colType = whereList[i].column_type
            ctype = to_ctype(colType)

            print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].index    = " + str(colIndex) + ";"

            if isinstance(conList[i], ystree.YFuncExp) and conList[i].func_name == "SUBQ":
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].relation = " + relList[i] + "_VEC;"
            elif relList[i] == "IN" and isinstance(conList[i], basestring):# in ("MOROCCO")
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].relation = EQ;"
            else:
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].relation = " + relList[i] + ";"
            print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].dataPos  = MEM;"

            # Get subquery result here
            if isinstance(conList[i], ystree.YFuncExp) and conList[i].func_name == "SUBQ":
                indexDict = {}
                for j in range(0, len(indexList)):
                    indexDict[ tn.table_name + "." + str(indexList[j])] = j
                generate_code_for_a_subquery(fo, lvl, relList[i], conList[i], "header.tupleNum", tnName, indexDict, tn, "MEM")

            if isinstance(conList[i], ystree.YFuncExp) and conList[i].func_name == "SUBQ":
                print >>fo, indent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &" + var_subqRes + ", sizeof(void *));"
            elif isinstance(conList[i], ystree.YFuncExp) and conList[i].func_name == "LIST":
                vec_len = len(conList[i].parameter_list)
                vec_item_len = type_length(whereList[i].table_name, whereList[i].column_name, whereList[i].column_type)
                print >>fo, indent + "(" + relName + ".filter)->exp[" + str(i) + "].vlen  = " + str(vec_len) + ";"
                print >>fo, indent + "{"
                print >>fo, indent + baseIndent + "char *vec = (char *)malloc(" + vec_item_len + " * " + str(vec_len) + ");"
                print >>fo, indent + baseIndent + "memset(vec, 0, " + vec_item_len + " * " + str(vec_len) + ");"
                for idx in range(0, vec_len):
                    item = conList[i].parameter_list[idx]
                    print >>fo, indent + baseIndent + "memcpy(vec + " + vec_item_len + " * " + str(idx) + ", " + item.cons_value + ", " + vec_item_len +");"
                print >>fo, indent + baseIndent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &vec, sizeof(char **));"
                print >>fo, indent + "}"
            elif ctype == "INT":
                if isinstance(conList[i], ystree.YRawColExp):
                    con_value = "*(int *)(_" + conList[i].table_name + "_" + str(conList[i].column_name) + ")"
                    # Get index here
                    if "--idx" in optimization:
                        indices.append(whereList[i].table_name.lower() + str(whereList[i].column_name) + "_idx")
                        los.append(conList[i].table_name.lower() + str(conList[i].column_name) + "_lo")
                        his.append(conList[i].table_name.lower() + str(conList[i].column_name) + "_hi")
                else:
                    con_value = conList[i]
                print >>fo, indent + "{"
                print >>fo, indent + baseIndent + "int tmp = " + con_value + ";"
                print >>fo, indent + baseIndent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp, sizeof(int));"
                print >>fo, indent + "}"
            elif ctype == "FLOAT":
                if isinstance(conList[i], ystree.YRawColExp):
                    con_value = "*(float *)(_" + conList[i].table_name + "_" + str(conList[i].column_name) + ")"
                else:
                    con_value = conList[i]
                print >>fo, indent + "{"
                print >>fo, indent + baseIndent + "float tmp = " + con_value + ";"
                print >>fo, indent + baseIndent + "memcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp, sizeof(float));"
                print >>fo, indent + "}"
            else:
                if isinstance(conList[i], ystree.YRawColExp):
                    con_value = "_" + conList[i].table_name + "_" + str(conList[i].column_name)
                else:
                    con_value = conList[i]
                print >>fo, indent + "strcpy((" + relName + ".filter)->exp[" + str(i) + "].content, " + con_value + ");\n"

        if CODETYPE == 0:
            # print >>fo, indent + resName + " = tableScan(&" + relName + ", &pp);"
            print >>fo, indent + "// space required on GPU: gpuFilter + gpuPsum + gpuCount + gpuExp + padding"
            print >>fo, indent + "size_t new_size = (gpu_inner_mp.free - gpu_inner_mp.base) + " + tnName + "->tupleNum * sizeof(int) + sizeof(int) * dim3(2048).x * dim3(256).x + sizeof(int) * dim3(2048).x * dim3(256).x + sizeof(struct whereExp) + 128;"
            print >>fo, indent + "resize_gpu_mempool(&gpu_inner_mp, new_size);"
            print >>fo, indent + "char *origin_pos = freepos_gpu_mempool(&gpu_inner_mp);"
            if len(indices) > 0:
                print >>fo, indent + resName + " = tableScan(&" + relName + ", &pp, true, true"  + (", true" if lvl > 0 else ", false") + ", " + indices[0] + ", " + los[0] + " + tupleid, " + his[0] + " + tupleid);"
            else:
                print >>fo, indent + resName + " = tableScan(&" + relName + ", &pp, true, true"  + (", true" if lvl > 0 else ", false") + ", NULL, NULL, NULL);"
            print >>fo, indent + "freeto_gpu_mempool(&gpu_inner_mp, origin_pos);"
        else:
            print >>fo, indent + resName + " = tableScan(&" + relName + ", &context, &pp);"

        print >>fo, indent + "clock_gettime(CLOCK_REALTIME, &diskStart);"
        if tablePreloaded:
            for i in range(0, totalAttr):
                print >>fo, indent + tnName + "->content[" + str(i) + "] = NULL;"

        #print >>fo, indent + "freeScan(&" + relName + ");\n"
        print >>fo, indent + "freeScan(&" + relName + ", false);\n"
        if CODETYPE == 1:
            print >>fo, indent + "clFinish(context.queue);"

        print >>fo, indent + "clock_gettime(CLOCK_REALTIME, &diskEnd);"
        # print >>fo, indent + "diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;"

        ############## end of wherecondition not none

    else:
        print >>fo, indent + resName + " = " + tnName + ";"

    print >>fo, indent + resName + "->colIdxNum = 0;"

    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}\n"

    return resName

def __traverse_tree(node, func):
    if isinstance(node, ystree.TwoJoinNode):
        __traverse_tree(node.left_child, func)
        __traverse_tree(node.right_child, func)
    elif isinstance(node, ystree.GroupByNode) or isinstance(node, ystree.OrderByNode) or isinstance(node, ystree.SelectProjectNode):
        __traverse_tree(node.child, func)

    func(node)

# check if a TableNode has new columns to load
def __gen_col_lists(tn, col_lists):
    if not isinstance(tn, ystree.TableNode):
        return

    indexList = []
    colList = []
    generate_col_list(tn, indexList, colList)

    if tn.table_name in col_lists:
        oldList = col_lists[tn.table_name]
        oldIndexList, oldColList = unmerge(oldList)
        for i in range(0, len(indexList)):
            if indexList[i] not in oldIndexList:
                oldIndexList.append(indexList[i])
                oldColList.append(colList[i])

        indexList = oldIndexList
        colList = oldColList

    col_lists[tn.table_name] = merge(indexList, colList)

def generate_col_lists(col_lists):
    return lambda tn: __gen_col_lists(tn, col_lists)


def generate_code_for_loading_tables(fo, indent, tree):
    global loaded_table_list
    global cols_to_index

    # traverse all query plan trees to find tables and their columns
    __traverse_tree(tree, generate_col_lists(loaded_table_list))
    map( lambda t: __traverse_tree(t, generate_col_lists(loaded_table_list)), ystree.subqueries )

    # calculate the table size (with loading columns)
    def cal_table_size(t_name, col_list):
        indexList, colList = unmerge(col_list)
        table_dir = "../../test/tables"
        column_files = [ table_dir + "/" + t_name + str(idx) for idx in indexList ]
        #print os.getcwd()
        return sum(map(lambda f: os.path.getsize(f), column_files))
        # return 0


    # extract which subqueries are in the next level
    def __extract_subquery_indices(node, indices):
        if node.where_condition is not None and node.where_condition.where_condition_exp is not None:
            where_exp = node.where_condition.where_condition_exp
            def __extract_subquery_indices_exp(exp, _indices):
                if isinstance(exp, ystree.YFuncExp) and exp.func_name == "SUBQ":
                    if exp.parameter_list[0] not in _indices:
                        _indices.append(exp.parameter_list[0].cons_value)

            def extract_subquery_indices_exp(_indices):
                return lambda e : __extract_subquery_indices_exp(e, _indices)

            ystree.__traverse_exp_map__(where_exp, extract_subquery_indices_exp(indices))

    def extract_subquery_indices(indices):
        return lambda n : __extract_subquery_indices(n, indices)

    # identify the levels of all subqueries
    indices = []
    subqs = []
    __traverse_tree(tree, extract_subquery_indices(indices))
    while len(indices) > 0:
        subqs.append( indices )
        new_indices = []
        map(lambda idx: __traverse_tree(ystree.subqueries[idx], extract_subquery_indices(new_indices)), indices)
        indices = new_indices

    # check if a table node is the requested table
    def __has_table(node, t_name, res_list):
        if not isinstance(node, ystree.TableNode):
            return

        if node.table_name == t_name:
            res_list.append(True)

    def has_table(t_name, res_list):
        return lambda n: __has_table(n, t_name, res_list)

    # get the deepest subquery level a table at
    def table_level(t_name, subqs):
        for i in range(len(subqs) - 1, -1, -1) :
            res = []
            for subq_idx in subqs[i]:
                __traverse_tree(ystree.subqueries[subq_idx], has_table(t_name, res))
                if len(res) > 0:
                    return i + 1
        return 0

    # sort all tables with their subquery level (descending) and their table size (ascending)
    # We remove tables depending on the subquery level first (a lower level first) and the table size (a larger table first)
    _loaded_list = [ (t_name, cal_table_size(t_name, c_list), table_level(t_name, subqs)) for t_name, c_list in loaded_table_list.items() ]
    _loaded_list.sort(key = lambda x: (x[2], -x[1]), reverse = True)

    # remove table until the preloaded tables are smaller than the threshold
    mem_size = sum( [ t[1] for t in _loaded_list ] )
    while mem_size > preload_threshold:
        t = _loaded_list.pop()
        mem_size -= t[1]
        del loaded_table_list[t[0]]


    def __extract_correlated_cols(node, col_list):
        if node.where_condition is not None and node.where_condition.where_condition_exp is not None:
            where_exp = node.where_condition.where_condition_exp
            is_passed_in_par = lambda par: isinstance(par, ystree.YConsExp) and par.ref_col is not None
            def __extract_correlated_cols_exp(exp, _col_list):
                if isinstance(exp, ystree.YFuncExp) and len(exp.parameter_list) == 2 and any(map(is_passed_in_par, exp.parameter_list)):
                    col = exp.parameter_list[1] if is_passed_in_par(exp.parameter_list[0]) else exp.parameter_list[0]
                    if isinstance(node, ystree.TableNode) and  isinstance(col, ystree.YRawColExp):
                        col_list.append(col.table_name + "." + str(col.column_name))

            def extract_correlated_cols_exp(_col_list):
                return lambda e: __extract_correlated_cols_exp(e, _col_list)

            ystree.__traverse_exp_map__(where_exp, extract_correlated_cols_exp(col_list))

    def extract_correlated_cols(col_list):
        return lambda n: __extract_correlated_cols(n, col_list)

    if optimization == "--idx":
        map( lambda t: __traverse_tree(t, extract_correlated_cols(cols_to_index)), ystree.subqueries )

    for t_name, c_list in loaded_table_list.items():
        generate_code_for_loading_a_table(fo, indent, t_name, c_list)


def generate_code_for_loading_a_table(fo, indent, t_name, c_list):

    tnName  = "_" + t_name.lower() + "_table"
    resName = t_name.lower() + "Table"

    indexList, colList = unmerge(c_list)

    sortList = map( lambda c: int(c.split(".")[1]), filter( lambda c: c.split(".")[0] == t_name, cols_to_index ) )


    print >>fo, indent + "// Load columns from the table " + t_name

    print >>fo, indent + "struct tableNode *" + resName + ";"
    # print >>fo, indent + resName + " = (struct tableNode *)malloc(sizeof(struct tableNode));"
    # print >>fo, indent + "CHECK_POINTER("+ resName + ");"
    print >>fo, indent + resName + " = (struct tableNode *)alloc_mempool(sizeof(struct tableNode));"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + "initTable(" + resName + ");"

    print >>fo, indent + "{"
    indent += baseIndent
    print >>fo, indent + "struct tableNode *" + tnName + ";"
    print >>fo, indent + "int outFd;"
    print >>fo, indent + "long outSize;"
    print >>fo, indent + "char *outTable;"
    print >>fo, indent + "long offset, tupleOffset;"
    print >>fo, indent + "int blockTotal;"
    print >>fo, indent + "struct columnHeader header;\n"

    totalAttr = len(indexList)
    if totalAttr <= 0:
        print "ERROR: Failed to generate code for loading the table " + t_name + ": no column is specified"
        exit(-1)

    firstTableFile = t_name + str(colList[0].column_name)
    print >>fo, indent + "// Retrieve the block number from " + firstTableFile
    print >>fo, indent + "outFd = open(\"" + firstTableFile + "\", O_RDONLY);"
    print >>fo, indent + "read(outFd, &header, sizeof(struct columnHeader));"
    print >>fo, indent + "blockTotal = header.blockTotal;"
    print >>fo, indent + "close(outFd);"
    print >>fo, indent + "offset = 0;"
    print >>fo, indent + "tupleOffset = 0;"

    print >>fo, indent + "for(int i = 0; i < blockTotal; i++){\n"
    indent += baseIndent
    print >>fo, indent + "// Table initialization"
    # print >>fo, indent + tnName + " = (struct tableNode *)malloc(sizeof(struct tableNode));"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + ");"
    # print >>fo, indent + tnName + "->totalAttr = " + str(totalAttr) + ";"
    # print >>fo, indent + tnName + "->attrType = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrType);"
    # print >>fo, indent + tnName + "->attrSize = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrSize);"
    # print >>fo, indent + tnName + "->attrIndex = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrIndex);"
    # print >>fo, indent + tnName + "->attrTotalSize = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->attrTotalSize);"
    # print >>fo, indent + tnName + "->dataPos = (int *)malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->dataPos);"
    # print >>fo, indent + tnName + "->dataFormat = (int *) malloc(sizeof(int) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->dataFormat);"
    # print >>fo, indent + tnName + "->content = (char **)malloc(sizeof(char *) * " + str(totalAttr) + ");"
    # print >>fo, indent + "CHECK_POINTER(" + tnName + "->content);\n"

    print >>fo, indent + tnName + " = (struct tableNode *)alloc_mempool(sizeof(struct tableNode));"
    print >>fo, indent + "MEMPOOL_CHECK();"
    print >>fo, indent + tnName + "->totalAttr = " + str(totalAttr) + ";"
    print >>fo, indent + tnName + "->attrType = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->attrSize = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->attrIndex = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->attrTotalSize = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->dataPos = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->dataFormat = (int *)alloc_mempool(sizeof(int) * " + str(totalAttr) + ");"
    print >>fo, indent + tnName + "->content = (char **)alloc_mempool(sizeof(char *) * " + str(totalAttr) + ");"
    print >>fo, indent + "MEMPOOL_CHECK();\n"


    tupleSize = "0"
    for i in range(0, totalAttr):
        col = colList[i]
        ctype = to_ctype(col.column_type)
        colIndex = int(col.column_name)
        colLen = type_length(t_name, colIndex, col.column_type)
        tupleSize += " + " + colLen

        print >>fo, indent + "// Load column " + str(colIndex) + ", type: " + col.column_type
        print >>fo, indent + tnName + "->attrSize[" + str(i) + "] = " + colLen + ";"
        print >>fo, indent + tnName + "->attrIndex["+ str(i) + "] = " + str(colIndex) + ";"
        print >>fo, indent + tnName + "->attrType[" + str(i) + "] = " + ctype + ";"

        if POS == 0:
            print >>fo, indent + tnName + "->dataPos[" + str(i) + "] = MEM;"
        elif POS == 1:
            print >>fo, indent + tnName + "->dataPos[" + str(i) + "] = PINNED;"
        elif POS == 2:
            print >>fo, indent + tnName + "->dataPos[" + str(i) + "] = UVA;"
        elif POS == 3:
            print >>fo, indent + tnName + "->dataPos[" + str(i) + "] = MMAP;"
        else:
            print >>fo, indent + tnName + "->dataPos[" + str(i) + "] = MEM;"

        print >>fo, indent + "outFd = open(\"" + t_name + str(colIndex) + "\", O_RDONLY);"
        print >>fo, indent + "offset = i * sizeof(struct columnHeader) + tupleOffset * " + str(colLen) + ";"
        print >>fo, indent + "lseek(outFd, offset, SEEK_SET);"
        print >>fo, indent + "read(outFd, &header, sizeof(struct columnHeader));"
        print >>fo, indent + "offset += sizeof(struct columnHeader);"
        print >>fo, indent + tnName + "->dataFormat[" + str(i) + "] = header.format;"
        print >>fo, indent + "outSize = header.tupleNum * " + colLen + ";"
        print >>fo, indent + tnName + "->attrTotalSize[" + str(i) + "] = outSize;\n"

        print >>fo, indent + "clock_gettime(CLOCK_REALTIME,&diskStart);"
        print >>fo, indent + "outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);"

        if CODETYPE == 0:
            if POS == 1:
                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)&" + tnName + "->content[" + str(i) + "], outSize));"
                print >>fo, indent + "memcpy(" + tnName + "->content[" + str(i) + "], outTable, outSize);"
            elif POS == 2:
                print >>fo, indent + "CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)&" + tnName+"->content["+str(i)+"], outSize));"
                print >>fo, indent + "memcpy(" + tnName + "->content[" + str(i) + "], outTable, outSize);"
            elif POS == 3:
                print >>fo, indent + tnName + "->content[" + str(i) + "] = (char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);"
            else:
                print >>fo, indent + tnName + "->content[" + str(i) + "] = (char *)memalign(256, outSize);"
                print >>fo, indent + "memcpy(" + tnName + "->content[" + str(i) + "], outTable, outSize);"
        else:
            if POS == 0:
                    print >>fo, indent + tnName + "->content[" + str(i) + "] = (char *)memalign(256, outSize);"
                    print >>fo, indent + "memcpy(" + tnName + "->content[" + str(i) + "], outTable, outSize);"
            elif POS == 3:
                    print >>fo, indent + tnName + "->content[" + str(i) + "] = (char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);"
            else:
                print >>fo, indent + tnName + "->content[" + str(i) + "] = (char *)clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, outSize, NULL, 0);"
                print >>fo, indent + "clTmp = clEnqueueMapBuffer(context.queue, (cl_mem)" + tnName + "->content[" + str(i) + "], CL_TRUE,CL_MAP_WRITE,0,outSize, 0, 0, 0, 0);"
                print >>fo, indent + "memcpy(clTmp, outTable, outSize);"
                print >>fo, indent + "clEnqueueUnmapMemObject(context.queue, (cl_mem)" + tnName + "->content[" + str(i) + "], clTmp, 0, 0, 0);"

        print >>fo, indent + "munmap(outTable, outSize);"
        print >>fo, indent + "clock_gettime(CLOCK_REALTIME, &diskEnd);"
        print >>fo, indent + "diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;"
        print >>fo, indent + "close(outFd);\n"

    print >>fo, indent + tnName + "->tupleSize = " + tupleSize + ";"
    print >>fo, indent + tnName + "->tupleNum = header.tupleNum;\n"

    print >>fo, indent + "if(blockTotal != 1){"

    if CODETYPE == 0:
        print >>fo, indent + baseIndent + "mergeIntoTable(" + resName + "," + tnName +", &pp);"
    else:
        print >>fo, indent + baseIndent + "mergeIntoTable(" + resName + "," + tnName +", &context, &pp);"

    print >>fo, indent + baseIndent + "clock_gettime(CLOCK_REALTIME, &diskStart);"
    print >>fo, indent + baseIndent + "freeTable(" + tnName + ");"
    if CODETYPE == 1:
        print >>fo, indent + baseIndent + "clFinish(context.queue);"

    print >>fo, indent + baseIndent + "clock_gettime(CLOCK_REALTIME, &diskEnd);"
    print >>fo, indent + baseIndent + "diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;"
    print >>fo, indent + "}else{"
    # print >>fo, indent + baseIndent + "free(" + resName + ");"
    print >>fo, indent + baseIndent + resName + " = " + tnName + ";"
    print >>fo, indent + "}"

    print >>fo, indent + "tupleOffset += header.tupleNum;"

    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}"

    print >>fo, indent + resName + "->colIdxNum = " + str(len(sortList)) + ";"
    print >>fo, indent + tnName + "->keepInGpuIdx = 1;"
    if len(sortList) > 0:
        # print >>fo, indent + resName + "->colIdx = (int *)malloc(sizeof(int) * " + str(len(sortList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + resName + "->colIdx);"
        # print >>fo, indent + resName + "->contentIdx = (int **)malloc(sizeof(int *) * " + str(len(sortList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + resName + "->contentIdx);"
        # print >>fo, indent + resName + "->posIdx = (int **)malloc(sizeof(int *) * " + str(len(sortList)) + ");"
        # print >>fo, indent + "CHECK_POINTER(" + resName + "->posIdx);\n"
        print >>fo, indent + resName + "->colIdx = (int *)alloc_mempool(sizeof(int) * " + str(len(sortList)) + ");"
        print >>fo, indent + resName + "->contentIdx = (int **)alloc_mempool(sizeof(int *) * " + str(len(sortList)) + ");"
        print >>fo, indent + resName + "->posIdx = (int **)alloc_mempool(sizeof(int *) * " + str(len(sortList)) + ");"
        print >>fo, indent + "MEMPOOL_CHECK();\n"

        for i in range(0, len(sortList)):
            print >>fo, indent + resName + "->colIdx[" + str(i) + "] = " + str( indexList.index(sortList[i]) ) + ";"
            print >>fo, indent + "createIndex(" + resName + ", " + str(i) + ", " + str( indexList.index(sortList[i]) ) +", &pp);\n"


    indent = indent[:indent.rfind(baseIndent)]
    print >>fo, indent + "}\n"

    return resName
