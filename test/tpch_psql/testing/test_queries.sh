#!/bin/bash

#Query dir
QUERY_DIR=/home/sofoklis/workspace/subquerygpu2/subquery/test/tpch_test


#Get results from PSQL
../deployment/psql testdb -f $QUERY_DIR/q2.sql > q2.output
#../deployment/psql testdb -f $QUERY_DIR/q4.sql > q4.output
#../deployment/psql testdb -f $QUERY_DIR/q16.sql > q16.output
#../deployment/psql testdb -f $QUERY_DIR/q18.sql > q18.output
