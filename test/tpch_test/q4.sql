--TPC-H Q4 (Type J - Original)
select
  *
from
  orders
where
  o_orderdate > '1996-01-15'
  and o_orderdate <= '1996-01-16'
;

-- select
--   o_orderpriority,
--   count(*) as order_count
-- from
--   orders
-- where
--   o_orderdate >= '1996-01-01'
--   and o_orderdate <= '1996-04-01'
--   and exists (
--     select
--       *
--     from
--       lineitem
--     where
--       l_orderkey = o_orderkey
--       and l_commitdate < l_receiptdate
--     )
-- group by
--   o_orderpriority
-- order by
--   o_orderpriority
-- ;
