select
  o_orderpriority,
  count(*) as order_count
from
  orders,
  (
    select
      l_orderkey
    from
      lineitem
    where
      l_commitdate < l_receiptdate
    group by
      l_orderkey
   ) l0
where
  o_orderdate > '1996-01-15'
  and o_orderdate <= '1996-04-15'
  and l0.l_orderkey = o_orderkey
group by
  o_orderpriority
-- order by
--   o_orderpriority
;

-- select
--   o_orderpriority,
--   count(*) as order_count
-- from
--   orders,
--   lineitem
-- where
--   o_orderdate > '1996-01-15'
--   and o_orderdate <= '1996-04-15'
--   and l_orderkey = o_orderkey
--   and l_commitdate < l_receiptdate
-- group by
--   o_orderpriority
-- -- order by
-- --   o_orderpriority
-- ;
