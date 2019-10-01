#!/usr/bin/python
"""
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import sys

global_table_dict = {}

class ColumnSchema:

    column_name = None
    column_type = None
    column_others = None

    table_schema = None

    def __init__(self, input_column_name, input_column_type):
        self.column_name = input_column_name
        self.column_type = input_column_type

    def setup_others(self, input_others):
        self.column_others = input_others

    def __repr__(self):
        return "[[ColumnSchema] name:" + self.column_name + ", type:" + self.column_type + "]"

class TableSchema:

    table_name = None
    column_list = None

    column_name_list = None

    def __init__(self, input_table_name, input_column_list):
        self.table_name = input_table_name
        self.column_list = input_column_list

        #let each column know where it is
        for ac in input_column_list:
            ac.table_schema = self

        self.column_name_list = []
        for a_column in self.column_list:
            self.column_name_list.append(a_column.column_name)


    def get_column_name_by_index(self, input_index):
        return self.column_list[input_index].column_name

    def get_column_type_by_index(self, input_index):
        return self.column_list[input_index].column_type

    def get_column_index_by_name(self, input_name):
        if input_name not in self.column_name_list:
            return -1
        else:
            return self.column_name_list.index(input_name)

    def get_column_type_by_name(self, input_name):
        index = self.get_column_index_by_name(input_name)
        if index != -1:
            return self.column_list[index].column_type
        else:
            return None

    def __repr__(self):
        return "[[TableSchema] name:" + self.table_name + ", column list:" + str(self.column_name_list) + "]"


def process_schema_in_a_file(schema_name):

    global global_table_dict

    #a line is a table, format: tablename, (columnname:columntype), ...
    fo = open(schema_name)
    als = fo.readlines()
    fo.close()

    for al in als:
        al = al.rstrip()

        t_al_a = al.split("|")

        table_name = t_al_a[0].upper()

        to_be_added_list = []
        for a_tmp_col_raw in t_al_a[1:]:
            if ":" not in a_tmp_col_raw:
                continue
            a_column_name = a_tmp_col_raw.split(":")[0].upper()
            a_column_type = a_tmp_col_raw.split(":")[1].upper()

            if a_column_type not in ["INTEGER", "DECIMAL", "TEXT", "DATE"]:
                print >>sys.stderr,"wrong column type:", a_column_type
                exit(39)

            a_col = ColumnSchema(a_column_name, a_column_type)

            if a_column_type in ["TEXT"]:
                a_column_length = a_tmp_col_raw.split(":")[2]
                a_col.setup_others(a_column_length)

            to_be_added_list.append(a_col)


        a_table = TableSchema(table_name, to_be_added_list)

        #print table_name, a_table
        global_table_dict[table_name] = a_table

    return global_table_dict

if __name__ == '__main__':

    process_schema_in_a_file(argv[1])
    print global_table_dict
