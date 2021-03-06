select 
  t1_sum_quantity,
  c_name,
  c_address,
  c_phone,
  n_name,
  r_name
from 
  orders, 
  customer,
  nation,
  region,
   (
     select
       sum(l_quantity) as t1_sum_quantity,
       l_orderkey as t1_orderkey
     from
       lineitem, part
     where
       p_partkey = l_partkey
       and p_size = 20
       and p_type like 'MEDIUM%'
       and p_brand = 'Brand#41'
     group by
       l_orderkey
   ) t1
where o_custkey = c_custkey
  and c_nationkey = n_nationkey
  and n_regionkey = r_regionkey 
  and o_orderkey = t1_orderkey
  and o_totalprice > 500000
  and 10 > t1_sum_quantity
;