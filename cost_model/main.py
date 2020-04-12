#!/usr/bin/python

import sys
import os
from schema import schema
from query import query
from config import config
from planner import planner

# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 3.0v 
# 
# This program estimates the execution of a nested/unnested query in GPU-DB.
#-------------------------------

# Main function
def main(argv):

	#Pick scaleFactor
	scaleFactor = 20

	#Create Schema (pass type and scale factor)
	s = schema('TPCH', scaleFactor)

	#Create query 
	#Test operators   : ('select', 'filter', 'join', 'aggregation', 'groupby')
	#Test complex ops : ('join2', 'join4', 'join4c4', 'join4c8')
	#Actual queries   : ('q2-nested', 'q2-unnested')
	q = query('join', scaleFactor )

	#Create system configuration ('TeslaV100' , 0)
	# Hardware: ('RI2')
	# GPU-DB Optimizations: 0 -> No opts, 1 -> mempool, 2 -> cache, 3-> All)
	c = config('RI2', 0)

	# #Planner
	p = planner (s, q, c)
	cost = p.estimateCost(True) # Estimate

	# #Print estimation (For single estimation)
	# planner.printRes() # Print final estimation 
	# planner.printResVerdose() # Print all estimations

	# #Forcast query (For multiple estimations with scale factors between 1 and 20)
	# for i in range (1,20):

	# 	#Set new scale factor
	# 	schema.setScaleFactor(i)

	# 	#Re-estimate
	# 	planner.estimateCost()

	# 	#Print result
	# 	print "Scale factor "+str(i)+" estimation is :"+planner.getEstimation()

	#Debug msg
	print "DONE."

	#Exit normally
	exit(0)

if __name__ == "__main__":
	main(sys.argv)