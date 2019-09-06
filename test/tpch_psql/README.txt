Instructions
-------------
1) Need to create a psql link from psql bin to this directory (default one is of an older version)

     ln -s /home/sofoklis/workspace/subquerygpu2/psql/bin-postgresql-11.5/bin/psql ./

2) In "deployment" directory, configure "PSQL_DIR" variable in "initdb.sh", "stardb.sh" and "stopdb.sh"

	This shuold be the psql installation (no need to change for DL190)
 
3) Init and start database using "initdb.sh", "stardb.sh". (Default name is "testdb")

4) Copy TPC-H data into the directory "load-data"

5) Load data in psql using 
    
      ./load-data/load_data.sh

6) Test tpc-h query 

     ./deployment/psql testdb -f ./q2.sql

8) Stop db using "stopdb.sh"
