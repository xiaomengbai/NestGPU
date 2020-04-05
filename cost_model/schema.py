
# Description 
#-------------------------------
#
# Author  : Sofoklis Floratos (floratos.1@osu.edu)
# Date    : 04/03/2020
# Version : 3.0v 
# 
# Class that captures the database schema to be used for the estimation.
#-------------------------------


#Class for SQL tables
class table:

	#Constructor
	def __init__(self, name, size, cols, scaleFactor=1):
		self.name = name 
		self.size = size 
		self.cols = cols
		self.scaleFactor = scaleFactor

	#Change scale factor
	def changeScaleFactor(self, scaleFactor):

		#Compute new table size
		base_size = self.size / self.scaleFactor # Go to scaleFactor of 1
		self.size = base_size * scaleFactor 

		#Update new scale factor
		self.scaleFactor = scaleFactor

#Class for SQL Schema
class schema:

	# Constructor
	def __init__(self, schemaName='TPCH', scaleFactor=1):

		#Check is schema config is supported
		if 'TPCH'.lower() == schemaName.lower():
			print "Creating TPC-H schema with scale factor of " + str (scaleFactor)
			self._createTpchSchema (scaleFactor)
		else:
			print "Schema "+schemaName+" is currently not available..."
			exit(1)

	# Create TPC-H schema
	def _createTpchSchema(self, scaleFactor):

		#Contains all the tables
		self.tables = {}

		# ----- Table "part" -----
		name = 'PART'
		size = 200000
		cols = {}
		cols['P_PARTKEY'] = 4
		cols['P_NAME'] = 55
		cols['P_MFGR'] = 25
		cols['P_BRAND'] = 10
		cols['P_TYPE'] = 25
		cols['P_SIZE'] = 4
		cols['P_CONTAINER'] = 10
		cols['P_RETAILPRICE'] = 4
		cols['P_COMMENT'] = 23
		self.tables['part'] = table(name, size, cols)

		# ----- Table "supplier" -----
		name = 'SUPPLIER'
		size = 10000
		cols = {}
		cols['S_SUPPKEY'] = 4
		cols['S_NAME'] = 25
		cols['S_ADDRESS'] = 40
		cols['S_NATIONKEY'] = 10
		cols['S_PHONE'] = 15
		cols['S_ACCTBAL'] = 4
		cols['S_COMMENT'] = 101
		self.tables['supplier'] = table(name, size, cols)

		# ----- Table "customer" -----
		name = 'CUSTOMER'
		size = 150000
		cols = {}
		cols['C_CUSTKEY'] = 4
		cols['C_NAME'] = 25
		cols['C_ADDRESS'] = 40
		cols['C_NATIONKEY'] = 10
		cols['C_PHONE'] = 15
		cols['C_ACCTBAL'] = 4
		cols['C_MKTSEGMENT'] = 10
		cols['C_COMMENT'] = 117
		self.tables['customer'] = table(name, size, cols)

		# ----- Table "nation" -----
		name = 'NATION'
		size = 25
		cols = {}
		cols['N_NATIONKEY'] = 4
		cols['N_NAME'] = 25
		cols['N_REGIONKEY'] = 4
		cols['N_COMMENT'] = 152
		self.tables['nation'] = table(name, size, cols)

		# ----- Table "region" -----
		name = 'REGION'
		size = 5
		cols = {}
		cols['R_REGIONKEY'] = 4
		cols['R_NAME'] = 25
		cols['R_COMMENT'] = 152
		self.tables['region'] = table(name, size, cols)

		# ----- Table "partsupp" -----
		name = 'PARTSUPP'
		size = 800000
		cols = {}
		cols['PS_PARTKEY'] = 4
		cols['PS_SUPPKEY'] = 4
		cols['PS_AVAILQTY'] = 4
		cols['PS_SUPPLYCOST'] = 4
		cols['PS_COMMENT'] = 199
		self.tables['partsupp'] = table(name, size, cols)

		# ----- Table "orders" -----
		name = 'ORDERS'
		size = 1500000
		cols = {}
		cols['O_ORDERKEY'] = 4
		cols['O_CUSTKEY'] = 4
		cols['O_ORDERSTATUS'] = 1
		cols['O_TOTALPRICE'] = 4
		cols['O_ORDERDATE'] = 10
		cols['O_ORDERPRIORITY'] = 15
		cols['O_CLERK'] = 15
		cols['O_SHIPPRIORITY'] = 4
		cols['O_COMMENT'] = 79
		self.tables['orders'] = table(name, size, cols)

		# ----- Table "lineitem" -----
		name = 'LINEITEM'
		size = 6000000
		cols = {}
		cols['L_ORDERKEY'] = 4
		cols['L_PARTKEY'] = 4
		cols['L_SUPPKEY'] = 4
		cols['L_LINENUMBER'] = 4
		cols['L_QUANTITY'] = 4
		cols['L_EXTENDEDPRICE'] = 4
		cols['L_DISCOUNT'] = 4
		cols['L_TAX'] = 4
		cols['L_RETURNFLAG'] = 1
		cols['L_LINESTATUS'] = 1
		cols['L_SHIPDATE'] = 10
		cols['L_COMMITDATE'] = 10
		cols['L_RECEIPTDATE'] = 10
		cols['L_SHIPINSTRUCT'] = 25
		cols['L_SHIPMODE'] = 10
		cols['L_COMMENT'] = 44
		self.tables['lineitem'] = table(name, size, cols)

		# Set scale factor
		self.setScaleFactor(scaleFactor)

	#Change scale factor of the datasete
	def setScaleFactor(self, scaleFactor):

		#Update tables
		for name in self.tables.keys():
			self.tables[name].changeScaleFactor(scaleFactor)

	#Print schema
	def printSchema(self):

		#Print all tables
		for table in self.tables.values():
			print "---Schema---"
			print " ---- "+table.name+" ---- "
			print "Tuples       :"+str(table.size)
			print "Cols Names   :"+str(table.cols.keys())
			print "Cols Size    :"+str(table.cols.values())
			print "Scale Factor :"+str(table.scaleFactor)
			print "-----------"

