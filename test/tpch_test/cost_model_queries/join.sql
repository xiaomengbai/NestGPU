SELECT p_mfgr 
FROM part, partsupp 
WHERE p_partkey = ps_partkey;