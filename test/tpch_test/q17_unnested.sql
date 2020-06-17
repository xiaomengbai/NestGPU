select
  sum(l_extendedprice * 0.1429) as avg_yearly
from
  lineitem,
  part,
  (
    select
      avg(l_quantity * 0.2) as avg_quantity, l_partkey as li0_partkey
    from
      lineitem
    group by
      l_partkey
  ) li0
where
  p_partkey = l_partkey
  and l_partkey = li0_partkey
  and p_brand = 'Brand#35'
  and p_container = 'WRAP CASE'
  and l_quantity < avg_quantity
;
