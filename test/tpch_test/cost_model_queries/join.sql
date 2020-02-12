select
   p_partkey,
   p_mfgr
from
   part,
   partsupp
where
   p_partkey = ps_partkey
;