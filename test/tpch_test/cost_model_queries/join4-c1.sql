select
   p_partkey
from
   part,
   partsupp,
   supplier,
   nation,
   region
where
   p_partkey = ps_partkey
   and s_suppkey = ps_suppkey
   and s_nationkey = n_nationkey
   and n_regionkey = r_regionkey
;

