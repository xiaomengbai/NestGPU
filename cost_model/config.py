
# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 4.0v 
# 
# Class that captures the configuration to be used for the estimation.
#-------------------------------


#Class for configuration
class config:


	# Constructor
	def __init__(self, gpu, optimizations):

		if gpu == "RI2":

			#Hardware and software configuration
			self.gpu = gpu
			self.optimizations = optimizations

			#Constant time
			self.constantTime = 12

			#Scan op configuration
			self.scan_diskTime = 1731343 #b/ms (~1.7 gb/sec for IR2)
			self.scan_KernelTime = 0
			self.scan_Threads = 1

			#Filter op configuration
			self.filter_kernelTime = 4.16
			self.filter_Threads = 524288

			#Join op configuration
			self.join_kernelTime_hashTable = 1.22
			self.join_kernelTime_probe = 3.7
			self.join_kernelTime_joinFact =  0.55 # (For 1 iteration, for col with 1 byte)
			self.join_kernelTime_joinDim = 0.167 # (For 1 iteration, for col with 1 byte)
			self.join_Threads = 1048576

			#Aggregation op configuraiton
			self.aggregation_kernelTime = 0.086
			self.aggregation_Threads = 131072

			#Group by op configurations
			self.groupby_kernelTime = 0.51
			self.groupby_Threads = 131072

			#Materialization time 
			self.materialization_kernelTime = 0.026 #(For 1 iteration copying 1 byte)
			self.materialization_threads = 65536



		else:
			print "Error - There is no existing configuration for this system!"
			exit(1)
		

	#Print out configuration
	def printConfig(self):

		print "---Config---"
		print "GPU           :"+self.gpu
		print "Optimizations :"+str(self.optimizations)
		print "ConstantTime  :"+str(self.constantTime)
		print "---Op Config---"
		print "Scan            : <"+str(self.scan_KernelTime)+" kernel time> <"+str(self.scan_Threads)+" gpu threads>"
		print "Filter          : <"+str(self.filter_kernelTime)+" kernel time> <"+str(self.filter_Threads)+" gpu threads>"
		print "Join op         : <"+str(self.join_kernelTime)+" kernel time> <"+str(self.join_Threads)+" gpu threads>"
		print "Aggregation     : <"+str(self.aggregation_kernelTime)+" kernel time> <"+str(self.aggregation_Threads)+" gpu threads>"
		print "Group By        : <"+str(self.groupby_kernelTime)+" kernel time> <"+str(self.groupby_Threads)+" gpu threads>"
		print "Materialization : <"+str(self.materialization_kernelTime)+" kernel time>"
		print "------------"
