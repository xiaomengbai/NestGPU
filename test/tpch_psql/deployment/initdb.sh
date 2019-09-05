#!/bin/bash

#Postgresql dir
PSQL_DIR=/home/sofoklis/workspace/subquerygpu2/psql

#Delete previous data folder
rm -r $PSQL_DIR/data-postgresql-11.5/
mkdir $PSQL_DIR/data-postgresql-11.5/

#Init directory
$PSQL_DIR/bin-postgresql-11.5/bin/initdb -D $PSQL_DIR/data-postgresql-11.5/

#Start db
$PSQL_DIR/bin-postgresql-11.5/bin/pg_ctl -D $PSQL_DIR/data-postgresql-11.5/ -l $PSQL_DIR/data-postgresql-11.5/logfile start

#Init database
$PSQL_DIR/bin-postgresql-11.5/bin/createdb testdb

#Stop db
$PSQL_DIR/bin-postgresql-11.5/bin/pg_ctl -D $PSQL_DIR/data-postgresql-11.5/ -l $PSQL_DIR/data-postgresql-11.5/logfile stop
