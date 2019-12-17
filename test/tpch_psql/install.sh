#!/bin/bash

if [ ! -e postgresql-12.1.tar.gz ]; then
    wget https://ftp.postgresql.org/pub/source/v12.1/postgresql-12.1.tar.gz;
fi
if [ ! -e postgresql-12.1/configure ]; then
    tar xfz postgresql-12.1.tar.gz;
fi
cd postgresql-12.1/
./configure --prefix=${PWD}
make -j8
make install
