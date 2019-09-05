#!/bin/bash
  
#Postgresql dir
PSQL_DIR=/home/sofoklis/workspace/subquerygpu2/psql

$PSQL_DIR/bin-postgresql-11.5/bin/pg_ctl -D $PSQL_DIR/data-postgresql-11.5/ -l $PSQL_DIR/data-postgresql-11.5/logfile stop
