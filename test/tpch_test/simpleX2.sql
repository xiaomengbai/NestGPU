select  p_mfgr
from part
where p_partkey = ( 
        select ps_partkey
        from partsupp
	where ps_suppkey = (
		select s_suppkey
		from supplier)
	)
;

-- select p_mfgr
-- from part, partsupp
-- where p_partkey = ps_partkey
-- ;
