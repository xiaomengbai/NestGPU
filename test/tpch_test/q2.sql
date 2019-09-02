--TPC-H Q2 (Type JA)
select
   s_acctbal,
   ps_suppkey,
   s_suppkey,
   p_partkey,
   ps_partkey,
   n_name,
   p_mfgr,
   s_address,
   s_phone,
   s_comment
from
      supplier,
      partsupp,
      part,
      nation,
      region
where
      s_suppkey = ps_suppkey
      and p_partkey = ps_partkey
      and p_size = 10
--    and p_type like '%[TYPE]'
--    and p_type like 'BURNISHED'
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'ASIA'
      and ps_supplycost = (
             select
                        min(ps_supplycost)
             from
                      partsupp, supplier,
                      nation, region
              where
                      p_partkey = ps_partkey
                      and s_suppkey = ps_suppkey
                      and s_nationkey = n_nationkey
                      and n_regionkey = r_regionkey
                      and r_name = 'ASIA'
              )
;


-- select max(ps_supplycost), ps_partkey
-- from partsupp, supplier, nation, region
-- where s_suppkey = ps_suppkey
-- and s_nationkey = n_nationkey
-- and n_regionkey = r_regionkey
-- and r_name = 'ASIA'
-- group by ps_partkey;


-- select min(ps_supplycost), ps_partkey
-- from partsupp--, supplier, nation, region
-- -- where s_suppkey = ps_suppkey
-- -- and s_nationkey = n_nationkey
-- -- and n_regionkey = r_regionkey
-- -- and r_name = 'ASIA'
-- where ps_partkey < 3
-- group by ps_partkey;




--TPC-H Q2 (Type JA)
-- select
--    s_acctbal,
--    ps_suppkey,
--    s_suppkey,
--    p_partkey,
--    ps_partkey,
--    n_name,
--    p_mfgr,
--    s_address,
--    s_phone,
--    s_comment
-- from
--       supplier,
--       partsupp,
--       part,
--       nation,
--       region
-- where
--        s_suppkey = ps_suppkey
--        and p_partkey = ps_partkey
--        and p_size = 10
-- --      and p_type like '%[TYPE]'
--       and s_nationkey = n_nationkey
--       and n_regionkey = r_regionkey
--       and r_name = 'ASIA'
--       and ps_supplycost = (
--              select
--                         min(ps_supplycost)
--              from
--                       partsupp, supplier,
--                       nation, region
--               where
--                       p_partkey = ps_partkey
--                       and s_suppkey = ps_suppkey
--                       and s_nationkey = n_nationkey
--                       and n_regionkey = r_regionkey
--                       and r_name = 'ASIA'
--               )
-- ;
