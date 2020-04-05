import math

# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 3.0v 
# 
# Class that captures the cost model to be used for the estimation.
#-------------------------------



#Class for cost estimation
class planner:


	# Constructor
	def __init__(self, schema, query, config):
		self.schema = schema
		self.query = query
		self.config = config

		#Keeps track of the working table size
		self.tmpTableSize = 0

	# Estimate disk time
	def _estimateDiskCost(self,kernel):

		#Compute data size that we need to fetch from disk
		row_size = 0
		col_size = 0
		for col in kernel.cols:
			col_size = self.schema.tables[kernel.table].cols[col.upper()]
			#print "Columm "+col+" with size " + str(col_size)+" added!"
			row_size += col_size
		total_data = row_size * self.schema.tables[kernel.table].size

		#Set size to full
		self.tmpTableSize = total_data

		#Compute and return time
		return self.tmpTableSize / self.config.scan_diskTime

	# Estimate filter time (i.e table scan)
	def _estimateFilterCost(self,kernel):

		#Compute number of iterations to be performed by the kernel
		iterations = self.schema.tables[kernel.table].size / self.config.filter_Threads
		iterations = math.ceil(iterations)

		#Compute data size that we need to process
		row_size = 0
		col_size = 0
		for col in kernel.cols:
			col_size = self.schema.tables[kernel.table].cols[col.upper()]
			#print "Columm "+col+" with size " + str(col_size)+" added!"
			row_size += col_size
		total_data = row_size * self.schema.tables[kernel.table].size

		#Calculate result size
		self.tmpTableSize = total_data * kernel.selectivity

		#Compute and return time
		return iterations * self.config.filter_kernelTime

	def _estimateJoinCost(self,kernel):

		#---------Build Hash table---------
		#We build the hash table on the right table
		hash_table_iterations = self.schema.tables[kernel.r_table].size / self.config.join_Threads
		hash_table_iterations = math.ceil(hash_table_iterations)

		#Compute hash table time
		build_hash_table_cost = hash_table_iterations * self.config.join_kernelTime_hashTable
		# print ">>Hash Table iter: "+str(hash_table_iterations)
		print ">>Hash Table time: "+str(build_hash_table_cost)
		#----------------------------------

		#---------Probe---------
		#We probe on the left table
		probe_iterations = self.schema.tables[kernel.l_table].size / self.config.join_Threads
		probe_iterations = math.ceil(probe_iterations)

		#Compute probe time
		probe_cost = probe_iterations * self.config.join_kernelTime_probe
		# print ">>Probe iter: "+str(probe_iterations)
		print ">>Probe time: "+str(probe_cost)
		#-----------------------

		#--Materialize Join result---

		#Compute data size that we need to process
		row_size = 0
		col_size = 0

		#Search all tables (ASSUME unique col names!)
		for table in self.schema.tables.keys():
			for col in kernel.cols:
				if col.upper() in self.schema.tables[table].cols:
					col_size = self.schema.tables[table].cols[col.upper()]
					row_size += col_size

		#Total data is the size of the row * result size of the join
		total_data = row_size * kernel.cardinality

		meterialization_cost = total_data / self.config.join_materialization_speed
		# print ">>Materialization data: "+str(total_data)
		print ">>Materialization time: "+str(meterialization_cost)
		#-----------------------------

		#Calculate result size
		self.tmpTableSize = total_data

		#Compute and return total time
		return build_hash_table_cost+probe_cost+meterialization_cost

	def _estimateAggCost(self,kernel):

		#Compute number of iterations to be performed by the kernel
		iterations = self.schema.tables[kernel.table].size / self.config.filter_Threads
		iterations = math.ceil(iterations)

		#Compute and return time
		return iterations * self.config.filter_kernelTime

	def _estimateGroupbyCost(self,kernel):
		return 24

	def _estimateMaterializationCost(self,kernel):

		#Compute and return time
		return self.tmpTableSize / self.config.materialization_kernelTime

	# Estimate execution time
	def estimateCost(self, verdose):

		#Compute un-nested part

		diskC = 0 #Disk cost
		filterC = 0 #Table scan cost
		joinC = 0 #Join cost
		aggregationC = 0 #aggregation cost
		groupbyC = 0 #group by cost

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
			print "Disk time          :"+str(diskC)
			print "Filter time        :"+str(filterC)
			print "Join time          :"+str(joinC)
			print "Aggregation time   :"+str(aggregationC)
			print "Group By time      :"+str(groupbyC)
			print "--Nested part--"
			print "Filter time (nest)      :"+str(nest_filterC)
			print "Join time (nest)        :"+str(nest_joinC)
			print "Aggregation time (nest) :"+str(nest_aggregationC)
			print "Group By time (next)    :"+str(nest_groupbyC)
			print "---------------"
			print "Total un-nested time :"+str(unnestedPartC)
			print "Total Nested time    :"+str(nestedPartC)
			print "---------------"
			print "Constant        time :"+str(self.config.constantTime)
			print "Materialization time :"+str(materializeC)
			print ">Total Time :"+str(total_cost)

		return total_cost
