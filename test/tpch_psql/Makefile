.PHONY: clean build

PSQL_DIR := postgresql-12.1
TABLES := part.tbl supplier.tbl partsupp.tbl customer.tbl nation.tbl region.tbl lineitem.tbl

help:
	@echo ""
	@echo "  build          download and build postgresql"
	@echo "  init-db        init a test database"
	@echo "  load-tables    load tables for TPC-H queries"
	@echo "  run-%          run %.sql"
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

run-%:
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile start
	$(PSQL_DIR)/bin/psql testdb -c '\timing' -f ../tpch_test/$*.sql > $*.output
	$(PSQL_DIR)/bin/pg_ctl -D $(PSQL_DIR)/data -l $(PSQL_DIR)/data/logfile stop

clean-tables:
	rm -f load_data/*.tbl

clean:
	rm -f postgresql-12.1.tar.gz *.output
	rm -r postgresql-12.1/
