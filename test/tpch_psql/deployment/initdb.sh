#!/bin/bash

#Postgresql dir
PSQL_DIR=/home/floratos.1/workspace/subquery/test/tpch_psql/postgresql-12.1

#Delete previous data folder
rm -r $PSQL_DIR/data
mkdir $PSQL_DIR/data

#Init directory
$PSQL_DIR/bin/initdb -D $PSQL_DIR/data

#Start db
$PSQL_DIR/bin/pg_ctl -D $PSQL_DIR/data -l $PSQL_DIR/data/logfile start

#Init database
$PSQL_DIR/bin/createdb testdb

#Stop db
$PSQL_DIR/bin/pg_ctl -D $PSQL_DIR/data -l $PSQL_DIR/data/logfile stop
