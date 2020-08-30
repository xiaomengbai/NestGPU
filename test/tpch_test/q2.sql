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
  region,
  nation,
  supplier,
  partsupp,
  part
  -- part,
  -- partsupp,
  -- supplier,
  -- nation,
  -- region
where
  p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 20
  and p_type like 'MEDIUM%'
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and ps_supplycost = (
    select
      min(ps_supplycost)
    from
      region, nation,
      supplier, partsupp
    where
       p_partkey = ps_partkey
       and s_suppkey = ps_suppkey
       and s_nationkey = n_nationkey
       and n_regionkey = r_regionkey
       and r_name = 'ASIA'
    )
order by
  s_acctbal,
  n_name,
  s_name,
  p_partkey
;
