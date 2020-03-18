select
  sum(l_extendedprice * 0.1429) as avg_yearly
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and p_brand = 'Brand#35'
  and p_container = 'WRAP CASE'
  and l_quantity < (
    select
      avg(l_quantity * 0.2)
    from
      lineitem
    where
      l_partkey = p_partkey
    )
;


