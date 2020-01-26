#!/usr/bin/python

import sys
import os
import string
import argparse
import math

#Description
#-------------------------------
# This is an script that generates TPC-H skewed data.
#
# To see the parameters use "-h" or "--help".
#
# Run script Example :
# python ./dbgen-skew.py -d ~/workspace/subquery/test/tables/ -t partsupp.tbl -c 2 -s 0.5
#-------------------------------

#Write data
def writeData(data, file):

    #Open file
    fstream = open(file, 'w+')

    #Write data
    for line in range (0, len(data)):
        for col in range (0, len(data[0])-1):
            fstream.write(data[line][col])
            fstream.write("|")
        fstream.write("\n")

    # Close file
    fstream.close()     

#Introduce Skew
def modifyData(data, col, skew):

    #Column index
    colIdx = int(col)-1

    #Calculate tuples need to be changes
    total = len(data)
    if (float(skew) < 1):
        skewed_tuples = int(math.floor(total * float(skew)))
    else:
        skewed_tuples = total

    #Get first value of column that need to have skew
    skew_value = data[0][colIdx]

    #Add skew
    for i in range (0,skewed_tuples):
        data[i][colIdx] = skew_value

    #Return data
    return data

#Read data from the TPC-H directory
def loadData(path, table):

    #CreateTable
    tableData = []

    #Open file
    fileName = path + "/" + table
    f = open(fileName, 'r')

    #Read file
    count = 0
    data = f.readlines()
    for line in range (0, len(data)):
        tableData.append(data[line].split('|'))
        count = count + 1

    #Print info
    print "Read "+str(count)+ " lines..."

    #Return array
    return tableData

# This function handles parameters
def handleArgs(currentPath):

    #Add parameter parsing
    parser = argparse.ArgumentParser(description='This is an script that generates TPC-H skewed data.')

    #Add data directory parameter
    parser.add_argument('-d', action="store", type=str, help='Path to the existing TPC-H data directory.')

    #Add data directory parameter
    parser.add_argument('-t', action="store", type=str, help='Name of the TPC-H table to generate.')

    #Add data directory parameter
    parser.add_argument('-c', action="store", type=str, help='Column(s) to introduce skew.', nargs='+')

    #Add data files sizes
    parser.add_argument('-s', action="store", type=str, help='Skew in each column.')

    #Parse arguments and return them
    return parser.parse_args()

# Main function
def main(argv):

    # Current Path
    currentPath = os.path.dirname(os.path.realpath(__file__))
    
    #Parse argument given by the user
    parsed_args = handleArgs(currentPath)

    #Configure output file
    path = os.path.dirname(os.path.realpath(__file__))
    outF = path + "/s"+parsed_args.s + "_" + parsed_args.t

    #Print info
    print "---- TPC-H Data Skew ----"
    print "Current TPC-H data direcotry : " + str(parsed_args.d)
    print "Name of the input table      : " + str(parsed_args.t)
    print "Columns processed            : " + str(parsed_args.c)
    print "Skew introduces              : " + str(parsed_args.s)
    print "Output file                  : " + str(outF)

    #Load data
    print "Loading data..."
    data = loadData(os.path.abspath(parsed_args.d), parsed_args.t)

    #Introduce Skew
    print "Introducing skew..."
    cols = parsed_args.c
    for i in range (0, len(cols)):
        skewed_data = modifyData(data, cols[i], parsed_args.s )

    #Write data
    print "Writing data..."
    writeData(skewed_data, outF)
    print "-------------------------"

if __name__ == "__main__":
    main(sys.argv)
