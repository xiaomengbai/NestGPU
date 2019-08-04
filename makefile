.PHONY: clean gpudb

help:
	@echo "help          print this message"
	@echo "driver        create $(CUDA_DRIVER)"
	@echo "gpudb         generate $(CUDA_GPUDB)"
	@echo "tables        generate tables ($(TABLES)) to $(DATA_DIR)"
	@echo "loader        generate the loader ($(LOADER))"
	@echo "load-columns  generate column files to $(DATA_DIR)"
	@echo "run           run the gpudb"
	@echo ""
	@echo "clean         clean all stuff"
	@echo "clean-gpudb   clean $(CUDA_DRIVER) and $(CUDA_GPUDB)"
	@echo ""

# build test/dbgen/dbgen for generating tables
DBGEN_DIR := test/dbgen
DBGEN := $(DBGEN_DIR)/dbgen
DBGEN_DIST ?= $(DBGEN_DIR)/dists.dss
TABLE_SCALE := 0.05
DBGEN_OPTS := -b $(DBGEN_DIST) -O hm -vfF -s $(TABLE_SCALE)

$(DBGEN):
	$(MAKE) -C $(DBGEN_DIR)


# target: tables
#   genrerate tables

# core dump when generating the *lineorder* table
# core dump when load the *date* table
TABLES := supplier customer part

DATA_DIR := test/tables
TABLE_FILES := $(foreach table,$(TABLES),$(DATA_DIR)/$(table).tbl)

tables: $(TABLE_FILES)

define GEN_TABLE_RULE
$$(DATA_DIR)/$(tname).tbl: $$(DBGEN) $$(DBGEN_DIST) $$(DATA_DIR)
	$$(DBGEN) $$(DBGEN_OPTS) -T $(shell echo $(tname) | head -c 1)
	mv $(tname).tbl $$@
endef

$(foreach tname,$(TABLES), \
    $(eval $(GEN_TABLE_RULE)) \
)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# target: loader
#   the program that loads columns from table files
UTIL_DIR := src/utility
LOADER_SRC := $(UTIL_DIR)/load.c
LOADER := $(UTIL_DIR)/gpuDBLoader

loader: $(LOADER)

$(LOADER): $(LOADER_SRC)
	$(MAKE) -C $(UTIL_DIR)

$(LOADER_SRC): $(SQL_FILE) $(SSB_SCHEMA) $(TRANSLATE_PY)
	python $(TRANSLATE_PY) $(SQL_FILE) $(SCHEMA_FILE)


# target: load-columns
#   generate column files from table files
LOAD_OPTS := --datadir $(DATA_DIR) $(foreach table,$(TABLES), --$(table) $(DATA_DIR)/$(table).tbl)
load-columns: tables loader
	$(LOADER) $(LOAD_OPTS)


# target: gpudb
#    generate and build the gpudb program

CUDA_DIR := src/cuda
CUDA_GPUDB := $(CUDA_DIR)/GPUDATABASE
CUDA_DRIVER := $(CUDA_DIR)/driver.cu

SSB_TEST_DIR := ./test/ssb_test
SSB_SCHEMA := $(SSB_TEST_DIR)/ssb.schema

SQL_FILE := test.sql
SCHEMA_FILE := $(SSB_SCHEMA)

TRANSLATE_PY := translate.py
CUDA_DRIVER_DEP := XML2CODE/code_gen.py XML2CODE/ystree.py

$(CUDA_DRIVER): $(SQL_FILE) $(SSB_SCHEMA) $(TRANSLATE_PY) $(CUDA_DRIVER_DEP)
	python $(TRANSLATE_PY) $(SQL_FILE) $(SCHEMA_FILE)

$(CUDA_GPUDB): $(CUDA_DRIVER)
	$(MAKE) -C $(CUDA_DIR)

driver: $(CUDA_DRIVER)
gpudb: $(CUDA_GPUDB)

run:
	$(CUDA_GPUDB) --datadir $(DATA_DIR)

clean-gpudb:
	$(MAKE) -C $(CUDA_DIR) clean
	rm -f $(CUDA_DRIVER)

clean: clean-gpudb
	$(MAKE) -C $(DBGEN_DIR) clean
	$(MAKE) -C $(UTIL_DIR) clean
	if [ -d $(DATA_DIR) ]; then \
	  rm -f $(DATA_DIR)/*; \
	  rm -r $(DATA_DIR); \
	fi
