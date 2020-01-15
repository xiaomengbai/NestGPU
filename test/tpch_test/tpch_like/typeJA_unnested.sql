-- Query (Type JA): TPC-H Q2 - No need for exaplanation. (Only avoid order by)
-- With this query we control the size of the outer table by changing p_size!

select
   s_acctbal,
   s_name,
   n_name,
   p_partkey,
   p_mfgr,
   s_address,
   s_phone,
   s_comment
 from
   part,
   partsupp,
   supplier,
   nation,
   region,
   (
     select
       min(ps_supplycost) as t1_min_supplycost,
       ps_partkey as t1_partkey
     from
       partsupp, supplier,
       nation, region
     where
       s_suppkey = ps_suppkey
       and s_nationkey = n_nationkey
       and n_regionkey = r_regionkey
       and r_name = 'ASIA'
     group by
       ps_partkey
   ) t1
 where
   p_partkey = ps_partkey
   and s_suppkey = ps_suppkey
   and p_size = 20
   and p_type like 'MEDIUM%'
   and s_nationkey = n_nationkey
   and n_regionkey = r_regionkey
   and r_name = 'ASIA'
   and ps_supplycost = t1_min_supplycost
   and p_partkey = t1_partkey
;

--Experiment -> Servey a p_size and p_type. Scale factor { 0.5, 1, 2, 4, 8 }
