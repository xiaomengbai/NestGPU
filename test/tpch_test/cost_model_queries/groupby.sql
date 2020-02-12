select
    min(ps_supplycost),
    ps_partkey
from
    partsupp
group by
       ps_partkey;