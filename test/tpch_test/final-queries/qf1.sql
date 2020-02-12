-- TPC-H Q2 (Type JA, Nested)
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
  nation
where
  p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 20
  and p_type like 'MEDIUM%'
  and p_brand = 'Brand#41'
  and s_nationkey = n_nationkey
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
