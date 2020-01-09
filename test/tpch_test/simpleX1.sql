select  p_mfgr
from part
where p_partkey = ( 
        select ps_partkey
        from partsupp)
;

-- select p_mfgr
-- from part, partsupp
-- where p_partkey = ps_partkey
-- ;
