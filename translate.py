#!/usr/bin/env python

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


import datetime
import os
import subprocess
import sys

sys.path.append("SQL2XML")
import sql2xml
sys.path.append("XML2CODE")
import code_gen

CURRENT_DIR = os.getcwd()
EXEC_DIR = 'bin'
TEMP_DIR = '.tmp'

def genXMLTree(queryFile, tmpFilePath):
    try:
        os.mkdir(TEMP_DIR)
    except:
        pass

    with open(queryFile) as inputFile:
        xmlStr = sql2xml.toXml(inputFile)
        with open(tmpFilePath, "w") as outputFile:
            outputFile.write(xmlStr)

def genGPUCode(schemaFile, tmpFilePath, optimization):
    
    #print 'TODO: call job generation program in ./bin/'
    os.chdir(CURRENT_DIR)

    if optimization is None:
        if tmpFilePath is None:
            cmd = 'python XML2CODE/main.py ' + schemaFile + ''
        else:
            cmd = 'python XML2CODE/main.py ' + schemaFile + ' ' + tmpFilePath + ''
    else:
            #Need to fix this one
            cmd = 'python XML2CODE/main.py ' + schemaFile + ' ' + tmpFilePath 
            #cmd = 'python XML2CODE/main.py ' + schemaFile + ' ' + tmpFilePath + ' ' + optimization
    
    subprocess.check_call(cmd, shell=True)

def print_usage():
    print 'usage 1: ./translate.py <schema-file>.schema'
    print 'usage 2: ./translate.py <query-file>.sql <schema-file>.schema'
    print 'usage 3: ./translate.py <query-file>.sql <schema-file>.schema --idx'

def main():

    if (len(sys.argv) != 2 and len(sys.argv) != 3 and len(sys.argv) != 4):
        print_usage()
        sys.exit(0)

    if len(sys.argv) == 3 or len(sys.argv) == 4:
        queryFile = sys.argv[1]
        schemaFile = sys.argv[2]
        tmpFile = str(datetime.datetime.now()).replace(' ', '_') + '.xml'
        tmpFilePath = './' + TEMP_DIR + '/' + tmpFile

        print '--------------------------------------------------------------------'
        print 'Generating XML tree (' + tmpFilePath + ')...'
        genXMLTree(queryFile, tmpFilePath)

        #Pass down optimization flag
        if len(sys.argv) == 4:
            optimization = sys.argv[3]
            print 'Generating optimized GPU Codes with options ['+optimization+']...'
            genGPUCode(schemaFile, tmpFilePath, optimization)
        else:
            print 'Generating GPU Codes ...'
            genGPUCode(schemaFile, tmpFilePath, None)

    else:
        queryFile = None
        schemaFile = sys.argv[1]
        genGPUCode(schemaFile, None, None)

    print 'Done'
    print '--------------------------------------------------------------------'
    # subprocess.check_call(['rm', '-rf', './' + TEMP_DIR])

    
if __name__ == "__main__":
    main()

