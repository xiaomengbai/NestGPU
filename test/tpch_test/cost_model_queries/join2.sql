select
   p_partkey,
   p_mfgr
from
   part,
   partsupp,
   supplier
where
   p_partkey = ps_partkey
   and s_suppkey = ps_suppkey
;