#!/bin/bash

# copy ystree_subq.py and code_gen_subq.py to XML2CODE/ and replace ystree.py and code_subq.py
./translate.py test.sql ./test/ssb_test/ssb.schema
