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

		#Search all tables (ASSUMPTION: unique col names!)
		for table in workspace.keys():
			if col in workspace[table].cols:
				return table

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
	def _estimateDiskCost(self,kernel):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols)

		#Get total data (always from scema as we load from disk!)
		total_data = row_size * self.schema.tables[kernel.table].size

		#Compute total time
		total_time = total_data / self.config.scan_diskTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.schema.tables[kernel.table].size)
		self.workSpace [kernel.output] = buffer

		#Return total time
		return total_time

	# Estimate filter time (i.e table scan)
	def _estimateFilterCost(self,kernel):

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

		#Return total time
		return total_time

	def _estimateJoinCost(self,kernel):

		#---------Build Hash table---------
		#We build the hash table on the right table
		hash_table_iterations = self.workSpace[kernel.r_table].size / self.config.join_Threads
		hash_table_iterations = math.ceil(hash_table_iterations)

		#Compute hash table time
		build_hash_table_cost = hash_table_iterations * self.config.join_kernelTime_hashTable
		print ">>Hash Table time: "+str(build_hash_table_cost)
		#----------------------------------

		#---------Probe---------
		#We probe on the left table
		probe_iterations = self.workSpace[kernel.l_table].size / self.config.join_Threads
		probe_iterations = math.ceil(probe_iterations)

		#Compute probe time
		probe_cost = probe_iterations * self.config.join_kernelTime_probe
		print ">>Probe time: "+str(probe_cost)
		#-----------------------

		#--Materialize part--

		#Join materialization total cost
		joinMaterializaiton_totalCost = 0 
		joinMaterializaiton_FactCost = 0
		joinMaterializaiton_DimCost = 0

		# Iterations for joinFact() and JoinDim() = Rows that qualify (i.e. cardinality ) / Join threads
		joinMaterializaiton_iterations = kernel.cardinality / self.config.join_Threads

		#For all columns 
		for out_col in kernel.output_cols:

			#If left table then compute joinFact
			if ( self._findTableFromCol( self.workSpace, out_col) == kernel.l_table ) :
				fact_cost = joinMaterializaiton_iterations *  self.config.join_kernelTime_joinFact * self._computeRowSize(self.schema, [out_col])
				joinMaterializaiton_FactCost += fact_cost
				joinMaterializaiton_totalCost += fact_cost
			
			#If right table them comput joinDim
			elif ( self._findTableFromCol( self.workSpace, out_col) == kernel.r_table ) :
				dim_cost = joinMaterializaiton_iterations *  self.config.join_kernelTime_joinDim * self._computeRowSize(self.schema, [out_col])
				joinMaterializaiton_DimCost += dim_cost
				joinMaterializaiton_totalCost += dim_cost

			#Else there is a problem
			else:
				print "Column "+out_col+" cannot be found in left or right table!"
				exit(0)

		print ">Left total materialization table cost (joinFact) :"+str(joinMaterializaiton_FactCost)
		print ">Right total materialization table cost (joinDim) :"+str(joinMaterializaiton_DimCost)
		#-----------------------------

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, kernel.cardinality)
		self.workSpace [kernel.output] = buffer

		#Compute and return total time
		return build_hash_table_cost + probe_cost + joinMaterializaiton_totalCost

	def _estimateAggCost(self,kernel):

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

		#Return total time
		return total_time

	def _estimateGroupbyCost(self,kernel):

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

		#Return total time
		return total_time

	def _estimateMaterializationCost(self,kernel):

		#Get input row size 
		row_size = self._computeRowSize(self.schema, kernel.cols) 
		
		#Get total data (always from the workSpace as we materialize mem buffers!)
		total_data = row_size * self.workSpace[kernel.table].size

		#Compute total time
		total_time = total_data / self.config.materialization_kernelTime

		#Add output to workspace
		buffer = tempTable(kernel.output, kernel.output_cols, self.workSpace[kernel.table].size)
		self.workSpace [kernel.output] = buffer

		#Return total time
		return total_time

	# Estimate execution time
	def estimateCost(self, verdose):

		#Create new schema that will hold only temp tables!
		self.workSpace = {}

		#Compute un-nested part
		diskC = 0 #Disk cost
		filterC = 0 #Table scan cost
		joinC = 0 #Join cost
		aggregationC = 0 #aggregation cost
		groupbyC = 0 #group by cost
		materializeC = 0 #materialization cost
		materializeJC = 0 #materialization after join cost

		for kernel in self.query.ops:

			if kernel.op == "scan":
				diskC += self._estimateDiskCost(kernel)

			elif kernel.op == "filter":
				filterC += self._estimateFilterCost(kernel)

			elif kernel.op == "join":
				joinC += self._estimateJoinCost(kernel)

			elif kernel.op == "aggregation":
				aggregationC += self._estimateAggCost(kernel)

			elif kernel.op == "group_by":
				groupbyC += self._estimateGroupbyCost(kernel)

			elif kernel.op == "materialize":
				materializeC = self._estimateMaterializationCost(kernel)

			else:
				print kernel.op + " should not be part of the unnested part."
				exit(1)

		unnestedPartC = diskC + filterC + joinC + aggregationC + groupbyC

		#Compute nested part
		nest_filterC = 0 #Table scan cost
		nest_joinC = 0 #Join cost
		nest_aggregationC = 0 #aggregation cost
		nest_groupbyC = 0 #group by cost

		for kernel in self.query.nestedOps:

			if  kernel.op == "filter":
				nest_filterC += self._estimateFilterCost(kernel)

			elif kernel.op == "join":
				nest_joinC += self._estimateJoinCost(kernel)

			elif kernel.op == "aggregation":
				nest_aggregationC += self._estimateAggCost(kernel)

			elif kernel.op == "group_by":
				nest_groupbyC += self._estimateGroupbyCost(kernel)

			else:
				print kernel.op + " should not be part of the nested part."
				exit(1)

		#Compute nested cost if there are no optimizations
		nestedPartC = 0
		if self.config.optimizations == 0:
			nestedPartC = (nest_filterC+nest_joinC+nest_aggregationC+nest_groupbyC) * self.query.outerTableRows
		elif self.config.optimizations > 0:
			print "Cost model cannot compute execution time with optimizations yet!"
			exit(1)

		#Linear part
		linear_part = unnestedPartC + nestedPartC

		#Total cost 
		total_cost = linear_part + self.config.constantTime + materializeC

		#Print detailed info
		if verdose == True:
			print "---Cost estimation---"
			print "Disk time               :"+str(diskC)
			print "Filter time             :"+str(filterC)
			print "Join time               :"+str(joinC)
			print "Aggregation time        :"+str(aggregationC)
			print "Group By time           :"+str(groupbyC)
			print "--Nested part--"
			print "Filter time (nest)      :"+str(nest_filterC)
			print "Join time (nest)        :"+str(nest_joinC)
			print "Aggregation time (nest) :"+str(nest_aggregationC)
			print "Group By time (next)    :"+str(nest_groupbyC)
			print "---------------"
			print "Total un-nested time    :"+str(unnestedPartC)
			print "Total Nested time       :"+str(nestedPartC)
			print "---------------"
			print "Constant           time :"+str(self.config.constantTime)
			print "Materialization    time :"+str(materializeC)
			print ">Total Time :"+str(total_cost)

		return total_cost
