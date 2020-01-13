--TPC-H Q4 (Type J - Original)
select
   o_orderpriority,
   count(*) as order_count 
from
   orders 
where
   o_orderdate >= 'date'
   and o_orderdate < date '[DATE]' + interval '3' month 
   and exists (
      select 
             *
       from 
             lineitem
       where
             l_orderkey = o_orderkey
             and l_commitdate < l_receiptdate
) 
group by
   o_orderpriority 
order by
   o_orderpriority;
