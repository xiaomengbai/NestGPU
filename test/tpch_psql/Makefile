.PHONY: clean build

PSQL_DIR := postgresql-12.1
TABLES := part.tbl supplier.tbl partsupp.tbl customer.tbl nation.tbl region.tbl

help:
	@echo ""
	@echo "  build          download and build postgresql"
	@echo "  init-db        init a test database"
	@echo "  load-tables    load tables for TPC-H queries"
	@echo "  run-q2         run q2.sql"
	@echo "  run-q2-unnest  run q2_unnested.sql"
	@echo "  run-qf3        run qf3.sql"
	@echo "  run-qf3-unnest run qf3_unnested.sql"
	@echo ""

build:
	if [ ! -e $(PSQL_DIR)/bin/psql ]; then \
	  ./install.sh; \
	fi

init-db:
	if [ ! -d $(PSQL_DIR)/data ]; then \
	  $(PSQL_DIR)/bin/initdb -D $(PSQL_DIR)/data; \
	  $(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start; \
	  $(PSQL_DIR)/bin/createdb testdb; \
	  $(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop; \
	fi

load-tables:
	for t in $(TABLES); do \
	  if [ ! -e load_data/$$t ]; then \
	    cp ../tables/$$t load_data/$$t; \
	  fi; \
	done
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -f load_data/load_schema.sql
	for t in $(TABLES); do \
	  table=$${t/.tbl/}; \
	  echo "Loading $$table ..."; \
	  sed 's/|$$//' load_data/$$t > /tmp/$$t; \
	  $(PSQL_DIR)/bin/psql testdb -q -c "TRUNCATE $$table"; \
	  $(PSQL_DIR)/bin/psql testdb -c "\\copy $$table FROM '/tmp/$$t' CSV DELIMITER '|'"; \
	done
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

run-q2:
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -f ../tpch_test/q2.sql > q2.output
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

run-q2-unnest:
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -f ../tpch_test/q2_unnested.sql > q2_unnested.output
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

run-qf3:
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -c '\timing' -f ../tpch_test/final-queries/qf3_mem.sql > qf3.output
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

run-qf3-unnest:
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -c '\timing' -f ../tpch_test/final-queries/qf3_mem_unnested.sql > qf3_unnested.output
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

clean:
	rm -f postgresql-12.1.tar.gz *.output
	rm -r postgresql-12.1/
