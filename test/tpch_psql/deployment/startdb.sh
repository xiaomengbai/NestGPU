#!/bin/bash
  
#Postgresql dir
PSQL_DIR=/home/floratos.1/workspace/subquery/test/tpch_psql/postgresql-12.1

$PSQL_DIR/bin/pg_ctl -D $PSQL_DIR/data -l $PSQL_DIR/data/logfile start
