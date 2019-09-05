#!/bin/bash

../deployment/psql testdb -f ./load_schema.sql 

for i in `ls *.tbl`; do
  table=${i/.tbl/}
  echo "Loading $table..."
  sed 's/|$//' $i > /tmp/$i
  ../deployment/psql testdb -q -c "TRUNCATE $table"
  ../deployment/psql testdb -c "\\copy $table FROM '/tmp/$i' CSV DELIMITER '|'"
done
