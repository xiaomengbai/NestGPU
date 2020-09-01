-- TPC-H Q2 (Type JA, Unnested)
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
   region,
   nation,
   supplier,
   partsupp,
   part,
   (
     select
       min(ps_supplycost) as t1_min_supplycost,
       ps_partkey as t1_partkey
     from
       region, nation,
       supplier, partsupp
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
   --and p_brand = 'Brand#41'
   --and p_brand like 'Brand#4%'
   and s_nationkey = n_nationkey
   and n_regionkey = r_regionkey
   and r_name = 'ASIA'
   and ps_supplycost = t1_min_supplycost
   and p_partkey = t1_partkey
--order by
--   s_acctbal,
--   n_name,
--   s_name,
--   p_partkey
;
