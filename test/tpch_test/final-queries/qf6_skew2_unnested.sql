--Search suppliers for part with special delivery that have not been used enough.

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
   part,
   partsupp,
   supplier,
   nation,
   (
     select
       min(s_acctbal) as t1_min_acctbal,
       s_suppkey as t1_suppkey
     from
      supplier,
      nation, 
      region
     where
        s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'ASIA'
     group by
       s_suppkey
   ) t1
 where
   p_partkey = ps_partkey
   and s_suppkey = ps_suppkey
   and p_size = 20
   and p_type like 'MEDIUM%'
   and p_brand = 'Brand#41'
   and s_nationkey = n_nationkey
   and s_acctbal = t1_min_acctbal
   and ps_suppkey = t1_suppkey
;