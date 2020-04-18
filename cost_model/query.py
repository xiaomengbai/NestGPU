
# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 4.0v 
# 
# Class that captures the database query to be used for the estimation.
#-------------------------------

#Class for materialize (i.e. time to convert col -> mem block)
class materialize:

	# Constructor
	def __init__(self, table, cols, output, output_cols):
		self.op = "materialize"
		self.table = table
		self.cols = cols
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Output      :"+ self.output
		print "Output Cols :"+ str(self.output_cols)

#Class for materialize after join (i.e. for some reason it takes more time to materialize after join)
class materializeJ:

	# Constructor
	def __init__(self, table, cols, output, output_cols):
		self.op = "materializeJ"
		self.table = table
		self.cols = cols
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Output      :"+ self.output
		print "Output Cols :"+ str(self.output_cols)

#Class for grouup by operation
class groupby:

	# Constructor
	def __init__(self, table, cols, expr, selectivity, output, output_cols):
		self.op = "group_by"
		self.table = table
		self.cols = cols
		self.expr = expr
		self.selectivity = selectivity
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Cardinality :"+ str(self.selectivity)
		print "Output      :"+ str(self.output)
		print "Output Cols :"+ str(self.output_cols)

#Class for aggregaton 
class aggregation:

	# Constructor
	def __init__(self, table, cols, expr, output, output_cols):
		self.op = "aggregation"
		self.table = table
		self.cols = cols
		self.expr = expr
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Output      :"+ str(self.output)
		print "Output Cols :"+ str(self.output_cols)

#Class for join (i.e. join)
class join:

	# Constructor
	def __init__(self, l_table, r_table, cols, expr, cardinality, output, output_cols):
		self.op = "join"
		self.l_table = l_table
		self.r_table = r_table
		self.cols = cols
		self.expr = expr
		self.cardinality = cardinality #For scale factor of 1
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Left table  :"+ self.l_table
		print "Right table :"+ self.r_table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Cardinality :"+ str(self.cardinality)
		print "Output      :"+ str(self.output)
		print "Output Cols :"+ str(self.output_cols)

#Class for filter (i.e. Table scan)
class filter:

	# Constructor
	def __init__(self, table, cols, expr, selectivity, output, output_cols):
		self.op = "filter"
		self.table = table
		self.cols = cols
		self.expr = expr
		self.selectivity = selectivity
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Expression  :"+ self.expr
		print "Selectivity :"+ str(self.selectivity)
		print "Output      :"+ str(self.output)
		print "Output Cols :"+ str(self.output_cols)

#Class for scan (i.e. Disk time)
class scan:

	# Constructor
	def __init__(self, table, cols, output, output_cols):
		self.op = "scan"
		self.table = table
		self.cols = cols
		self.output = output
		self.output_cols = output_cols

	#Pring op
	def printOp(self):
		print "Operator    :"+ self.op
		print "Table       :"+ self.table
		print "Cols        :"+ str(self.cols)
		print "Output      :"+ self.output
		print "Output Cols :"+ str(self.output_cols)

#Class for SQL Schema
class query:

	# Constructor
	def __init__(self, query, scaleFactor):

		#Configure simple ops queries
		if 'select'.lower() == query.lower():

			#Name of the test case
			self.name = 'select'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr'],\
				'select-tmp1', ['p_partkey', 'p_mfgr']))
			self.ops.append(materialize ('select-tmp1', ['p_partkey', 'p_mfgr'],\
				'select-tmp2', ['p_partkey', 'p_mfgr'] ))

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
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr', 'p_brand'],\
				'filter-tmp1', ['p_partkey', 'p_mfgr', 'p_brand']))
			self.ops.append(filter ('filter-tmp1', ['p_partkey', 'p_mfgr'], "p_brand like Brand4%", 0.196,\
				'filter-tmp2', ['p_partkey', 'p_mfgr']))
			self.ops.append(materialize ('filter-tmp2', ['p_partkey', 'p_mfgr'],\
				'filter-tmp3', ['p_partkey', 'p_mfgr'] ))

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
			self.ops.append(scan ('partsupp', ['ps_supplycost'],\
				'agg-tmp1', ['ps_supplycost']))
			self.ops.append(aggregation ('agg-tmp1', ['ps_supplycost'], "min(ps_supplycost)",\
				'agg-tmp2', ['ps_supplycost']))
			self.ops.append(materialize ('agg-tmp2', ['ps_supplycost'],\
				'agg-tmp3', ['ps_supplycost']))

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
			self.ops.append(scan ('partsupp', ['ps_supplycost','ps_partkey'],\
				'groupby-tmp1', ['ps_supplycost', 'ps_partkey']))
			self.ops.append(groupby ('groupby-tmp1', ['ps_supplycost','ps_partkey'], "min(ps_supplycost) GROUP BY ps_partkey", 0.25,\
				'groupby-tmp2', ['ps_supplycost', 'ps_partkey'] ))
			self.ops.append(materialize ('groupby-tmp2', ['ps_supplycost', 'ps_partkey'],\
				'groupby-tmp3', ['ps_supplycost', 'ps_partkey']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join'.lower() == query.lower():

			#Name of the test case
			self.name = 'join'

			#Actual SQL query
			self.sql = "SELECT p_mfgr FROM part, partsupp WHERE p_partkey = ps_partkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr'],\
				'join-tmp1', ['p_partkey', 'p_mfgr'] ))
			self.ops.append(scan ('partsupp', ['ps_partkey'],\
				'join-tmp2', ['ps_partkey']))
			
			#Join part and partsupp
			self.ops.append(join ("join-tmp1", "join-tmp2", ['p_partkey', 'ps_partkey'], "p_partkey = ps_partkey", 800000 * scaleFactor,\
				'join-tmp3', ['p_mfgr']))
			
			#Materialize res
			self.ops.append(materialize ('join-tmp3', ['p_mfgr'],\
				'join-tmp4', ['p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'joinc2'.lower() == query.lower():

			#Name of the test case
			self.name = 'joinc2'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part, partsupp WHERE p_partkey = ps_partkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr'],\
				'join-tmp1', ['p_partkey', 'p_mfgr'] ))
			self.ops.append(scan ('partsupp', ['ps_partkey'],\
				'join-tmp2', ['ps_partkey']))
			
			#Join part and partsupp
			self.ops.append(join ("join-tmp1", "join-tmp2", ['p_partkey', 'ps_partkey'], "p_partkey = ps_partkey", 800000 * scaleFactor,\
				'join-tmp3', ['p_partkey', 'p_mfgr']))
			
			#Materialize res
			self.ops.append(materialize ('join-tmp3', ['p_partkey', 'p_mfgr'],\
				'join-tmp4', ['p_partkey', 'p_mfgr']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'joinc4'.lower() == query.lower():

			#Name of the test case
			self.name = 'joinc4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr, p_name, ps_comment FROM part, partsupp WHERE p_partkey = ps_partkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr', 'p_name'],\
				'join-tmp1', ['p_partkey', 'p_mfgr', 'p_name'] ))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_comment'],\
				'join-tmp2', ['ps_partkey','ps_comment']))
			
			#Join part and partsupp
			self.ops.append(join ("join-tmp1", "join-tmp2", ['p_partkey', 'ps_partkey'], "p_partkey = ps_partkey", 800000 * scaleFactor,\
				'join-tmp3', ['p_partkey', 'p_mfgr', 'p_name', 'ps_comment']))
			
			#Materialize res
			self.ops.append(materialize ('join-tmp3', ['p_partkey', 'p_mfgr', 'p_name', 'ps_comment'],\
				'join-tmp4', ['p_partkey', 'p_mfgr', 'p_name', 'ps_comment']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join2c2'.lower() == query.lower():

			#Name of the test case
			self.name = 'join2'

			#Actual SQL query
			self.sql = "SELECT p_partkey, p_mfgr FROM part, partsupp, supplier WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey;"

			#Kernels
			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr'], \
				'join2-tmp1', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey'],\
				'join2-tmp2', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey'],\
				'join2-tmp3', ['s_suppkey']))

			#Join part and supplier
			self.ops.append(join ('join2-tmp1', 'join2-tmp2', ['p_partkey', 'ps_partkey'], "p_partkey = ps_partkey", 800000 * scaleFactor,\
				'join2-tmp4', ['p_partkey', 'p_mfgr', 'ps_suppkey']))
			
			#Join 2 part-partsupp with supplier
			self.ops.append(join ('join2-tmp4', "join2-tmp3", ['s_suppkey', 'ps_suppkey'], "s_suppkey = ps_suppkey", 800000 * scaleFactor,\
				'join2-tmp5', ['p_partkey', 'p_mfgr'] ))

			#Materialize res
			self.ops.append(materialize ('join2-tmp5', ['p_partkey', 'p_mfgr'],\
				'join2-tmp6', ['p_partkey', 'p_mfgr'] ))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join4c2'.lower() == query.lower():

			#Name of the test case
			self.name = 'join2c4'

			#Actual SQL query
			self.sql = "SELECT p_partkey, s_name FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey'],\
				'join4c8-tmp1', ['p_partkey']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey'],\
				'join4c8-tmp2', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey','s_name', 's_nationkey'],\
				'join4c8-tmp3', ['s_suppkey', 's_name', 's_nationkey']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey'],\
				'join4c8-tmp4', ['n_nationkey','n_regionkey']))
			self.ops.append(scan ('region', ['r_regionkey'],\
				'join4c8-tmp5', ['r_regionkey']))

			#Join part and partsupp
			self.ops.append(join ('join4c8-tmp1', 'join4c8-tmp2', ['p_partkey', 'ps_partkey'] , "p_partkey = ps_partkey", 800000 * scaleFactor, \
				'join4c8-tmp6', ['p_partkey', 'ps_suppkey']))
			
			#Join2 part-partsupp and supplier
			self.ops.append(join ("join4c8-tmp6", "join4c8-tmp3", ['s_suppkey', 'ps_suppkey'], "s_suppkey = ps_suppkey", 800000 * scaleFactor, \
				'join4c8-tmp7', ['p_partkey', 's_name', 's_nationkey']))

			#Join3 part-partsupp-supplier and nation
			self.ops.append(join ("join4c8-tmp7", "join4c8-tmp4", ['s_nationkey', 'n_nationkey'], "s_nationkey = n_nationkey", 800000 * scaleFactor, \
				'join4c8-tmp8', ['p_partkey', 's_name', 'n_regionkey']))			

			#Join4 part-partsupp-supplier and nation
			self.ops.append(join ("join4c8-tmp8", "join4c8-tmp5", ['n_regionkey', 'r_regionkey'], "n_regionkey = r_regionkey", 800000 * scaleFactor, \
				'join4c8-tmp9', ['p_partkey', 's_name']))			

			#Print result
			self.ops.append(materialize ('join4c8-tmp9', ['p_partkey', 's_name'],\
				'join4c8-tmp10', ['p_partkey', 's_name']))

			#Nested part
			self.nestedOps = []
			self.outerTableRows = 0

		elif 'join4c8'.lower() == query.lower():

			#Name of the test case
			self.name = 'join4c8'

			#Actual SQL query
			self.sql = "SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment FROM part, partsupp, supplier, nation, region WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey;"

			self.ops = [] 
			self.ops.append(scan ('part', ['p_partkey', 'p_mfgr'],\
				'join4c8-tmp1', ['p_partkey', 'p_mfgr']))
			self.ops.append(scan ('partsupp', ['ps_partkey', 'ps_suppkey'],\
				'join4c8-tmp2', ['ps_partkey', 'ps_suppkey']))
			self.ops.append(scan ('supplier', ['s_suppkey', 's_acctbal', 's_name', 's_address', 's_phone', 's_nationkey', 's_comment'],\
				'join4c8-tmp3', ['s_suppkey','s_acctbal','s_name','s_address', 's_phone','s_nationkey','s_comment']))
			self.ops.append(scan ('nation', ['n_nationkey','n_regionkey', 'n_nationkey', 'n_name'],\
				'join4c8-tmp4', ['n_nationkey','n_regionkey','n_name']))
			self.ops.append(scan ('region', ['r_regionkey'],\
				'join4c8-tmp5', ['r_regionkey']))

			#Join part and partsupp
			self.ops.append(join ('join4c8-tmp1', 'join4c8-tmp2', ['p_partkey', 'ps_partkey'] , "p_partkey = ps_partkey", 800000 * scaleFactor, \
				'join4c8-tmp6', ['p_partkey', 'p_mfgr', 'ps_suppkey']))
			
			#Join2 part-partsupp and supplier
			self.ops.append(join ("join4c8-tmp6", "join4c8-tmp3", ['s_suppkey', 'ps_suppkey'], "s_suppkey = ps_suppkey", 800000 * scaleFactor, \
				'join4c8-tmp7', ['p_partkey', 'p_mfgr', 's_acctbal','s_name','s_address', 's_phone', 's_nationkey', 's_comment']))

			#Join3 part-partsupp-supplier and nation
			self.ops.append(join ("join4c8-tmp7", "join4c8-tmp4", ['s_nationkey', 'n_nationkey'], "s_nationkey = n_nationkey", 800000 * scaleFactor, \
				'join4c8-tmp8', ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name','n_regionkey']))			

			#Join4 part-partsupp-supplier and nation
			self.ops.append(join ("join4c8-tmp8", "join4c8-tmp5", ['n_regionkey', 'r_regionkey'], "n_regionkey = r_regionkey", 800000 * scaleFactor, \
				'join4c8-tmp9', ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name']))			

			#Print result
			self.ops.append(materialize ('join4c8-tmp9', ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name'],\
				'join4c8-tmp10', ['p_partkey', 'p_mfgr','s_acctbal','s_name','s_address', 's_phone','s_comment','n_name']))

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

