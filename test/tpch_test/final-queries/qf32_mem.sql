select 
  c_name,
  c_address,
  c_phone,
  n_name,
  r_name
from 
  orders, 
  customer,
  nation,
  region
where o_custkey = c_custkey
  and c_nationkey = n_nationkey
  and n_regionkey = r_regionkey 
  and o_totalprice > 500000 
  and 10 > (
    select sum(l_quantity)
    from lineitem, part
    where o_orderkey = l_orderkey
      and l_partkey = p_partkey
      and p_size = 20
      and p_type like 'MEDIUM%'
      and p_brand = 'Brand#41'
  );
