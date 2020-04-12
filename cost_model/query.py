
# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 3.0v 
# 
# Class that captures the database query to be used for the estimation.
#-------------------------------



#Class for materialize (i.e. time to convert col -> mem block)
class materialize:

	# Constructor
	def __init__(self, table, cols):
		self.op = "materialize"
		self.table = table
		self.cols = cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)

#Class for grouup by operation
class groupby:

	# Constructor
	def __init__(self, table, cols, expr, selectivity):
		self.op = "group_by"
		self.table = table
		self.cols = cols
		self.expr = expr
		self.selectivity = selectivity

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Cardinality :"+ str(self.selectivity)

#Class for aggregaton 
class aggregation:

	# Constructor
	def __init__(self, table, cols, expr):
		self.op = "aggregation"
		self.table = table
		self.cols = cols
		self.expr = expr

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr

#Class for join (i.e. join)
class join:

	# Constructor
	def __init__(self, l_table, r_table, cols, expr, cardinality):
		self.op = "join"
		self.l_table = l_table
		self.r_table = r_table
		self.cols = cols
		self.expr = expr
		self.cardinality = cardinality #For scale factor of 1

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Left table  :"+ self.l_table
		print "Right table :"+ self.r_table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Cardinality :"+ str(self.cardinality)

#Class for filter (i.e. Table scan)
class filter:

	# Constructor
	def __init__(self, table, cols, expr, selectivity):
		self.op = "filter"
		self.table = table
		self.cols = cols
		self.expr = expr
		self.selectivity = selectivity

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Selectivity :"+ str(self.selectivity)

#Class for scan (i.e. Disk time)
class scan:

	# Constructor
	def __init__(self, table, cols):
		self.op = "scan"
		self.table = table
		self.cols = cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)

#Class for SQL Schema
class query:

	# Constructor
	def __init__(self, query):

		#Configure simple ops queries
		if 'select'.lower() == query.lower():

			#Name of the test case
			self.name = 'select'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(materialize ('part', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'filter'.lower() == query.lower():

			#Name of the test case
			self.name = 'filter'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part WHERE p_brand like Brand4%;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr', 'p_brand']))
			self.ops.append(filter ('part', ['p_partkey', 'p_mfgr'], "p_brand like Brand4%", 0.196))
			self.ops.append(materialize ('part', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join'.lower() == query.lower():

			#Name of the test case
			self.name = 'join'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part, partsupp WHERE p_partkey = ps_partkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr'], "p_partkey = ps_partkey", 800000))

			self.ops.append(materialize ('tmp', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'aggregation'.lower() == query.lower():

			#Name of the test case
			self.name = 'aggregation'

			#Actual SQL query
			self.sql = "SELECT min(ps_supplycost) FROM partsupp;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('partsupp', ['ps_supplycost']))
			self.ops.append(aggregation ('partsupp', ['ps_supplycost'], "min(ps_supplycost)"))
			self.ops.append(materialize ('partsupp', ['ps_supplycost']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'groupby'.lower() == query.lower():

			#Name of the test case
			self.name = 'groupby'

			#Actual SQL query
			self.sql = "SELECT min(ps_supplycost),ps_partkey FROM partsupp GROUP BY ps_partkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('partsupp', ['ps_supplycost','ps_partkey']))
			self.ops.append(groupby ('partsupp', ['ps_supplycost','ps_partkey'], "min(ps_supplycost) GROUP BY ps_partkey", 0.25))
			self.ops.append(materialize ('partsupp', ['ps_supplycost','ps_partkey']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join2'.lower() == query.lower():

			#Name of the test case
			self.name = 'join2'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part, partsupp, supplier WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr'], "p_partkey = ps_partkey", 800000))
			
			#self.ops.append(join ("tmp", "supplier", ['p_partkey', 'p_mfgr'], "s_suppkey = ps_suppkey", 800000))
			self.ops.append(join ("partsupp", "supplier", ['p_partkey', 'p_mfgr'], "s_suppkey = ps_suppkey", 800000))

			self.ops.append(materialize ('tmp1', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join4'.lower() == query.lower():

			#Name of the test case
			self.name = 'join4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey']))
			self.ops.append(scan ('region', ['r_regionkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr'], "p_partkey = ps_partkey", 800000))
			
			# self.ops.append(join ("tmp", "supplier", ['p_partkey', 'p_mfgr'], "s_suppkey = ps_suppkey", 800000))
			# self.ops.append(join ("tmp1", "nation", ['p_partkey', 'p_mfgr'], "s_nationkey = n_nationkey", 800000))
			# self.ops.append(join ("tmp3", "region", ['p_partkey', 'p_mfgr'], "n_regionkey = r_regionkey", 800000))
			
			self.ops.append(join ("partsupp", "supplier", ['p_partkey', 'p_mfgr'], "s_suppkey = ps_suppkey", 800000))
			self.ops.append(join ("partsupp", "nation", ['p_partkey', 'p_mfgr'], "s_nationkey = n_nationkey", 800000))
			self.ops.append(join ("partsupp", "region", ['p_partkey', 'p_mfgr'], "n_regionkey = r_regionkey", 800000))

			self.ops.append(materialize ('tmp1', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join4c4'.lower() == query.lower():

			#Name of the test case
			self.name = 'join4c4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr, s_acctbal FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey','s_acctbal']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey']))
			self.ops.append(scan ('region', ['r_regionkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr'], "p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey", 800000))

			#Name of the test case
			self.name = 'join4c4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr, s_acctbal, s_name FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey','s_acctbal','s_name']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey']))
			self.ops.append(scan ('region', ['r_regionkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "p_partkey = ps_partkey", 800000))
			# self.ops.append(join ("tmp", "supplier", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "s_suppkey = ps_suppkey", 800000))
			# self.ops.append(join ("tmp1", "nation", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "s_nationkey = n_nationkey", 800000))
			# self.ops.append(join ("tmp2", "region", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "n_regionkey = r_regionkey", 800000))

			self.ops.append(join ("partsupp", "supplier", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "s_suppkey = ps_suppkey", 800000))
			self.ops.append(join ("partsupp", "nation", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "s_nationkey = n_nationkey", 800000))
			self.ops.append(join ("partsupp", "region", ['p_partkey', 'p_mfgr','s_acctbal','s_name'], "n_regionkey = r_regionkey", 800000))


			self.ops.append(materialize ('tmp3', ['p_partkey', 'p_mfgr','s_acctbal','s_name']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join4c8'.lower() == query.lower():

			#Name of the test case
			self.name = 'join4c4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr, s_acctbal, s_name FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey','s_acctbal','s_name','s_address', 's_phone','s_comment']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey','n_name']))
			self.ops.append(scan ('region', ['r_regionkey']))

			self.ops.append(join ("part", "partsupp", ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name'], "p_partkey = ps_partkey", 800000))
			self.ops.append(join ("tmp", "supplier", ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name'], "s_suppkey = ps_suppkey", 800000))
			self.ops.append(join ("tmp1", "nation", ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name'], "s_nationkey = n_nationkey", 800000))
			self.ops.append(join ("tmp2", "region", ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name'], "n_regionkey = r_regionkey", 800000))

			self.ops.append(materialize ('tmp3', ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'q2-nested'.lower() == query.lower():

			self.op = 'q2-nested'

			#Nested part
			self.nestedOps = [] 

		elif 'q2-unnested'.lower() == query.lower():
			self.op = 'q2-unnested'

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		else:
			print "Query "+schemaName+" is currently not available..."
			exit(1)

	def printQuery(self):
		print "---Query---"
		print "Name :"+self.name
		print "SQL  :"+self.sql
		print "<< Query Plan >>"
		for op in self.ops:
			op.printOp()
			print "-"
		print "-----------"

