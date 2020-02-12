select
   p_partkey,
   p_mfgr
from
   part
where
    p_type like 'MEDIUM%';