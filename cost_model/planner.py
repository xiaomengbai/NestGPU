import math

# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 4.0v 
# 
# Class that captures the cost model to be used for the estimation.
#-------------------------------

#Class that represents temporary tables
class tempTable:

	#Constructor
	def __init__(self, name, cols, size):
		self.name = name
		self.cols = cols
		self.size = size

#Class that contains results for nested and un-nested kernels
class resultQuery():

	#Constructor
	def __init__(self, resUnnested, resNested, totalTime):
		self.unnestedPart = resUnnested
		self.nestedPart = resNested
		self.totalQueryTime = totalTime

	#Print estimation
	def printRes(self):

		print "<---Unnested Part--->"
		self.unnestedPart.printRes()
		print ""
		print "<----Nested Part---->"
		self.nestedPart.printRes()
		print ""
		print "<------------------->"
		print ">>Total estimated time:"+ str(self.totalQueryTime)

	#Print estimation with more details
	def printResVerdose(self):

		print "<---Unnested Part--->"
		self.unnestedPart.printResVerdose()
		print "<----Nested Part---->"
		self.nestedPart.printResVerdose()
		print "<------------------->"
		print ">>Total estimated time:"+ str(self.totalQueryTime)

#Class that holds all the estimated results for kernels
class resultEstimation:

	#Constructor
	def __init__(self):

		#Keep total and per kernel estimation
		self.total_diskTime = 0
		self.diskTime = []

		self.total_filterTime = 0
		self.filterTime = []

		self.total_aggregationTime = 0
		self.aggregationTime = []

		self.total_groupbyTime = 0
		self.groupbyTime = []

		# -- Join Estimations --
		self.total_hashTableTime = 0
		self.hashTableTime = []

		self.total_probeTime = 0
		self.probeTime = []

		self.total_joinFactTime = 0
		self.joinFactTime = []

		self.total_joinDimTime = 0
		self.joinDimTime = []

		self.total_joinTime = 0
		self.joinTime = []
		# --

		self.total_constantTime = 0

		#Note: Used only in the unnested part as 
		#      we materialize only at the very end.
		self.total_materializationTime = 0
		self.materializationTime = []

		#Total execution time
		self.total_Time = 0

	#Compute total time
	def estimateTotalTime(self):
		self.total_Time = self.total_constantTime + \
			self.total_diskTime + \
			self.total_filterTime + \
			self.total_aggregationTime + \
			self.total_groupbyTime + \
			self.total_joinTime + \
			self.total_materializationTime

	#Print results
	def printRes(self):
		print "Total Time :"+str(self.total_Time)
		print "> Cosntant Time    :"+str(self.total_constantTime)
		print ""
		print "> Disk Time        :"+str(self.total_diskTime)
		print "> Filter Time      :"+str(self.total_filterTime) 
		print "> Aggregation Time :"+str(self.total_aggregationTime) 
		print "> GroupBy Time     :"+str(self.total_groupbyTime) 
		print ""
		print "> Join Time        :"+str(self.total_joinTime) 
		print ">> HashTable :"+str(self.total_hashTableTime) 
		print ">> Probe     :"+str(self.total_probeTime) 
		print ">> JoinFact  :"+str(self.total_joinFactTime) 
		print ">> JoinDim   :"+str(self.total_joinDimTime) 
		print ""
		print "> Materialization  :"+str(self.total_materializationTime)

	#Print results
	def printResVerdose(self):
		print "Total Time :"+str(self.total_Time)
		print "> Cosntant Time    :"+str(self.total_constantTime)
		print ""
		print "> Disk Time        :"+str(self.total_diskTime)
		print ">>> :"+str(self.diskTime)
		print ""
		print "> Filter Time      :"+str(self.total_filterTime) 
		print ">>> :"+str(self.filterTime)
		print ""
		print "> Aggregation Time :"+str(self.total_aggregationTime) 
		print ">>> :"+str(self.aggregationTime)
		print ""
		print "> GroupBy Time     :"+str(self.total_groupbyTime) 
		print ">>> :"+str(self.groupbyTime)
		print ""
		print "> Join Time        :"+str(self.total_joinTime) 
		print ">>> :"+str(self.joinTime)
		print ""
		print ">> HashTable :"+str(self.total_hashTableTime) 
		print ">>>>> :"+str(self.hashTableTime)
		print ""
		print ">> Probe     :"+str(self.total_probeTime) 
		print ">>>>> :"+str(self.probeTime)
		print ""
		print ">> JoinFact  :"+str(self.total_joinFactTime) 
		print ">>>>> :"+str(self.joinFactTime)
		print ""
		print ">> JoinDim   :"+str(self.total_joinDimTime) 
		print ">>>>> :"+str(self.joinDimTime)
		print ""
		print "> Materialization  :"+str(self.total_materializationTime)
		print ">>> :"+str(self.materializationTime)

#Class for cost estimation
class planner:

	# Constructor
	def __init__(self, schema, query, config):
		self.schema = schema
		self.query = query
		self.config = config

		#Keeps track of the working table size
		self.tmpTableSize = 0

	# Find the table in which the column belongs
	def _findTableFromCol(self, workspace, col):

		#Find all temp tables that have this column
		temp_tables = []

		#Search all tables (ASSUMPTION: unique col names!)
		for table in workspace.keys():
			if col in workspace[table].cols:
				temp_tables.append(table)

		return temp_tables

	# Compute size of a row given cols
	def _computeRowSize(self, schema, cols):

		row_size = 0
		col_size = 0

		#Search all tables (ASSUMPTION: unique col names!)
		for table in schema.tables.keys():
				for col in cols:
					if col.upper() in schema.tables[table].cols:
						col_size = schema.tables[table].cols[col.upper()]
						row_size += col_size

		#Return size
		return row_size

	# Estimate disk time
	def _estimateDiskCost(self, kernel, res):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols)

		#Get total data (always from scema as we load from disk!)
		total_data = row_size * self.schema.tables[kernel.table].size

		#Compute total time
		total_time = total_data / self.config.scan_diskTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.schema.tables[kernel.table].size)
		self.workSpace [kernel.output] = buffer

		#Add kernel total time to the result 
		res.diskTime.append(total_time)
		res.total_diskTime += total_time

	# Estimate filter time (i.e table scan)
	def _estimateFilterCost(self, kernel, res):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols) 
		
		#Get total data (always from the workSpace as we materialize mem buffers!)
		total_data = row_size * self.workSpace[kernel.table].size

		#Compute number of iterations to be performed by the kernel
		iterations = self.workSpace[kernel.table].size / self.config.filter_Threads
		iterations = math.ceil(iterations)

		#Compute total time
		total_time = iterations * self.config.filter_kernelTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.workSpace[kernel.table].size * kernel.selectivity)
		self.workSpace [kernel.output] = buffer

		#Add kernel total time to the result 
		res.filterTime.append(total_time)
		res.total_filterTime += total_time

	# Estimate join time (i.e. hash Join)
	def _estimateJoinCost(self, kernel, res):

		#Keeps track of the total join time
		total_joinTime = 0

		#---------Build Hash table---------
		#We build the hash table on the right table
		hash_table_iterations = self.workSpace[kernel.r_table].size / self.config.join_Threads
		hash_table_iterations = math.ceil(hash_table_iterations)

		#Compute hash table time
		build_hash_table_cost = hash_table_iterations * self.config.join_kernelTime_hashTable
		
		#Add hash table total time to the result 
		res.hashTableTime.append(build_hash_table_cost)
		res.total_hashTableTime += build_hash_table_cost
		
		#Add time to the estimation of the current join
		total_joinTime += build_hash_table_cost
		#----------------------------------

		#----------------Probe-------------
		#We probe on the left table
		probe_iterations = self.workSpace[kernel.l_table].size / self.config.join_Threads
		probe_iterations = math.ceil(probe_iterations)

		#Compute probe time
		probe_cost = probe_iterations * self.config.join_kernelTime_probe

		#Add probe total time to the result 
		res.probeTime.append(probe_cost)
		res.total_probeTime += probe_cost

		#Add time to the estimation of the current join
		total_joinTime += probe_cost
		#---------------------------------

		#------------Materialize part------------

		# Iterations for joinFact() and JoinDim() = Rows that qualify (i.e. cardinality ) / Join threads
		joinMaterializaiton_iterations = kernel.cardinality / self.config.join_Threads

		#For all columns 
		for out_col in kernel.output_cols:

			#If left table then compute joinFact
			if ( kernel.l_table in self._findTableFromCol( self.workSpace, out_col) ) :
				
				#Compute fact materialization cost
				fact_cost = joinMaterializaiton_iterations *  self.config.join_kernelTime_joinFact * self._computeRowSize(self.schema, [out_col])

				#Add time to the result
				res.joinFactTime.append(fact_cost)
				res.total_joinFactTime += fact_cost

				#Add time to the estimation of the current join
				total_joinTime += fact_cost
			
			#If right table them comput joinDim
			elif ( kernel.r_table in self._findTableFromCol( self.workSpace, out_col)) :
				
				#Compute dim materialization cost
				dim_cost = joinMaterializaiton_iterations *  self.config.join_kernelTime_joinDim * self._computeRowSize(self.schema, [out_col])

				#Add time to the result
				res.joinDimTime.append(dim_cost)
				res.total_joinDimTime += dim_cost

				#Add time to the estimation of the current join
				total_joinTime += dim_cost

			#Else there is a problem
			else:
				print "Column "+out_col+" cannot be found in left or right table!"
				exit(0)
		#-----------------------------

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, kernel.cardinality)
		self.workSpace [kernel.output] = buffer

		#Add time to the result
		res.joinTime.append(total_joinTime)
		res.total_joinTime += total_joinTime

	# Estimate aggregation time (i.e. aggregation without groupby)
	def _estimateAggCost(self, kernel, res):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols) 
		
		#Get total data (always from the workSpace as we materialize mem buffers!)
		total_data = row_size * self.workSpace[kernel.table].size

		#Compute number of iterations to be performed by the kernel
		iterations = self.workSpace[kernel.table].size / self.config.aggregation_Threads
		iterations = math.ceil(iterations)

		#Compute total time
		total_time = iterations * self.config.aggregation_kernelTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, 1)
		self.workSpace [kernel.output] = buffer

		#Add kernel total time to the result 
		res.aggregationTime.append(total_time)
		res.total_aggregationTime += total_time

	# Estimate group by time (i.e. aggregation with groupby)
	def _estimateGroupbyCost(self, kernel, res):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols) 
		
		#Get total data (always from the workSpace as we materialize mem buffers!)
		total_data = row_size * self.workSpace[kernel.table].size

		#Compute number of iterations to be performed by the kernel
		iterations = self.workSpace[kernel.table].size / self.config.groupby_Threads
		iterations = math.ceil(iterations)

		#Compute total time
		total_time = iterations * self.config.groupby_kernelTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.workSpace[kernel.table].size * kernel.selectivity)
		self.workSpace [kernel.output] = buffer

		#Add kernel total time to the result 
		res.groupbyTime.append(total_time)
		res.total_groupbyTime += total_time

	# Estimate materialization time
	def _estimateMaterializationCost(self, kernel, res):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols) 
		
		#Compute number of iterations to be performed by the kernel
		iterations = self.workSpace[kernel.table].size / self.config.materialization_threads
		iterations = math.ceil(iterations)

		#Compute total time
		total_time = iterations * self.config.materialization_kernelTime * row_size

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.workSpace[kernel.table].size)
		self.workSpace [kernel.output] = buffer

		#Add kernel total time to the result 
		res.materializationTime.append(total_time)
		res.total_materializationTime += total_time

	# Estimate execution time
	def estimateCost(self):

		#Create new schema that will hold only temp tables!
		self.workSpace = {}

		#Object to keep the results
		resUnnested = resultEstimation()

		#Add constant time 
		resUnnested.total_constantTime = self.config.constantTime

		#Compute time for unnested kernels
		for kernel in self.query.ops:

			if kernel.op == "scan":
				self._estimateDiskCost(kernel,resUnnested)

			elif kernel.op == "filter":
				self._estimateFilterCost(kernel,resUnnested)

			elif kernel.op == "join":
				self._estimateJoinCost(kernel,resUnnested)

			elif kernel.op == "aggregation":
				self._estimateAggCost(kernel,resUnnested)

			elif kernel.op == "group_by":
				self._estimateGroupbyCost(kernel,resUnnested)

			elif kernel.op == "materialize":
				self._estimateMaterializationCost(kernel,resUnnested)

			else:
				print kernel.op + " should not be part of the unnested part."
				exit(1)

		#Compute total time for unnested part
		resUnnested.estimateTotalTime()

		#Same for nested
		resNested = resultEstimation()

		#Add constant time
		resUnnested.total_constantTime = self.config.constantTime

		#Compute time for nested kernels
		for kernel in self.query.nestedOps:

			if  kernel.op == "filter":
				self._estimateFilterCost(kernel,resNested)

			elif kernel.op == "join":
				self._estimateJoinCost(kernel,resNested)

			elif kernel.op == "aggregation":
				self._estimateAggCost(kernel,resNested)

			elif kernel.op == "group_by":
				self._estimateGroupbyCost(kernel,resNested)

			else:
				print kernel.op + " should not be part of the nested part."
				exit(1)

		#Compute total time for nested part
		resNested.estimateTotalTime()

		#Compute nested cost if there are no optimizations
		totalTime = 0
		if self.config.optimizations == 0:
			totalTime =  (resNested.total_Time * self.query.outerTableRows) + resUnnested.total_Time
		elif self.config.optimizations > 0:
			print "Cost model cannot compute execution time with optimizations yet!"
			exit(1)

		#Pack everything to a query result and return it
		return resultQuery (resUnnested, resNested, totalTime)