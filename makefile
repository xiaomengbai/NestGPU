.PHONY: help tables loader load-columns driver gpudb run clean clean-gpudb clean-loader clean-all

help:
	@echo "help          print this message"
	@echo "driver        create $(CUDA_DRIVER)"
	@echo "gpudb         generate $(CUDA_GPUDB)"
	@echo "tables        generate tables ($(TABLES)) to $(DATA_DIR)"
	@echo "loader        generate the loader ($(LOADER))"
	@echo "load-columns  generate column files to $(DATA_DIR)"
	@echo "run           run the gpudb"
	@echo ""
	@echo "clean"
	@echo "clean-gpudb   clean $(CUDA_DRIVER) and $(CUDA_GPUDB)"
	@echo ""
	@echo "clean-all     clean all stuff"


SSB_TEST_DIR := ./test/ssb_test
SSB_SCHEMA := $(SSB_TEST_DIR)/ssb.schema
TPCH_TEST_DIR := ./test/tpch_test
TPCH_SCHEMA := $(TPCH_TEST_DIR)/tpch.schema

SCHEMA := $(TPCH_SCHEMA)

# --- Queries ---
# Testing query
#SQL_FILE := $(TPCH_TEST_DIR)/test.sql

# TPC-H Q2
#SQL_FILE := $(TPCH_TEST_DIR)/q2.sql
SQL_FILE := $(TPCH_TEST_DIR)/q2_simple.sql
#SQL_FILE := $(TPCH_TEST_DIR)/q2_unnested.sql

# TPC-H Q4
#SQL_FILE := $(TPCH_TEST_DIR)/q4.sql

# TPC-H Q16
#SQL_FILE := $(TPCH_TEST_DIR)/q16.sql

# TPC-H Q18
#SQL_FILE := $(TPCH_TEST_DIR)/q18.sql
# ---------------


# -- Optimizations --

#Baseline
GPU_OPT := --base

#Nested indexes (sort and prefix)
#GPU_OPT := --idx
# -------------------

# build test/dbgen/dbgen for generating tables
DBGEN_DIR := test/dbgen
DBGEN := $(DBGEN_DIR)/dbgen
DBGEN_DIST ?= $(DBGEN_DIR)/dists.dss
TABLE_SCALE := 0.75
#index works for 0.01, 0.1, 0.5, 1 (out of memory)
DBGEN_OPTS := -b $(DBGEN_DIST) -O hm -vfF -s $(TABLE_SCALE)

$(DBGEN):
	$(MAKE) -C $(DBGEN_DIR)

# target: tables
#   genrerate tables
TABLES := supplier part customer nation region partsupp

DATA_DIR := test/tables
TABLE_FILES := $(foreach table,$(TABLES),$(DATA_DIR)/$(table).tbl)

tables: $(TABLE_FILES)

define GEN_TABLE_RULE
$$(DATA_DIR)/$(tname).tbl: $$(DBGEN) $$(DBGEN_DIST)
	mkdir -p $$(DATA_DIR)
	$$(DBGEN) $$(DBGEN_OPTS) -T $(shell echo $(tname) | head -c 1)
	mv $(tname).tbl $$@
endef

$(foreach tname,$(filter-out part,$(TABLES)), \
    $(eval $(GEN_TABLE_RULE)) \
)

$(DATA_DIR)/part.tbl: $(DBGEN) $(DBGEN_DIST)
	mkdir -p $(DATA_DIR)
	$(DBGEN) $(DBGEN_OPTS) -T p
	mv part.tbl $(DATA_DIR)/part.tbl
	mv partsupp.tbl $(DATA_DIR)/partsupp.tbl

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# target: loader
#   the program that loads columns from table files
UTIL_DIR      := src/utility
INC_DIR       := src/include

LOADER        := $(UTIL_DIR)/gpuDBLoader

LOADER_SRC    := $(UTIL_DIR)/load.c
SCHEMA_H      := $(INC_DIR)/schema.h

CONFIG_PY     := XML2CODE/config.py
SCHEMA_PY     := XML2CODE/schema.py
LOADER_GEN_PY := XML2CODE/loader_gen.py
SCHEMA_GEN_PY := XML2CODE/schema_gen.py


loader: $(LOADER)

$(LOADER): $(LOADER_SRC) $(SCHEMA_H)
	$(MAKE) -C $(UTIL_DIR)

$(LOADER_SRC): $(LOADER_GEN_PY) $(SCHEMA) $(SCHEMA_PY) $(CONFIG_PY)
	python $(LOADER_GEN_PY) $(SCHEMA)

$(SCHEMA_H): $(SCHEMA_GEN_PY) $(SCHEMA) $(SCHEMA_PY) $(CONFIG_PY)
	python $(SCHEMA_GEN_PY) $(SCHEMA)

# target: load-columns
#   generate column files from table files
LOAD_OPTS := --datadir $(DATA_DIR) $(foreach table,$(TABLES), --$(table) $(DATA_DIR)/$(table).tbl)
load-columns: $(TABLE_FILES) $(LOADER)
	$(LOADER) $(LOAD_OPTS)


# target: gpudb
#    generate and build the gpudb program

CUDA_DIR        := src/cuda
CUDA_GPUDB      := $(CUDA_DIR)/GPUDATABASE
CUDA_DRIVER     := $(CUDA_DIR)/driver.cu

TRANSLATE_PY    := translate.py
CUDA_DRIVER_DEP := XML2CODE/code_gen.py \
                   XML2CODE/ystree.py   \
                   XML2CODE/schema.py   \
                   XML2CODE/config.py

SELF := makefile
$(CUDA_DRIVER): $(SQL_FILE) $(SCHEMA) $(TRANSLATE_PY) $(CUDA_DRIVER_DEP) $(SELF)
	python $(TRANSLATE_PY) $(SQL_FILE) $(SCHEMA) $(GPU_OPT)

driver: $(CUDA_DRIVER)
gpudb: driver
	$(MAKE) -C $(CUDA_DIR)

run: gpudb
	$(CUDA_GPUDB) --datadir $(DATA_DIR)

# clean
clean: clean-gpudb
clean-gpudb:
	$(MAKE) -C $(CUDA_DIR) clean
	rm -f $(CUDA_DRIVER)

clean-loader:
	$(MAKE) -C $(UTIL_DIR) clean
	rm -f $(LOADER_SRC)
	rm -f $(SCHEMA_H)

clean-all: clean-gpudb clean-loader
	$(MAKE) -C $(DBGEN_DIR) clean
	if [ -d $(DATA_DIR) ]; then \
	  rm -f $(DATA_DIR)/*; \
	  rm -r $(DATA_DIR); \
	fi
