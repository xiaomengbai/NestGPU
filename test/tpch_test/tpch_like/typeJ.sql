-- Query (Type J): Find all orders within a month that had more than one item.
-- With this query we control the size of the outer table.
-- Similar query used in SIGMOD PAPER : "A nested relational approach to processing SQL subqueries." 
SELECT O_ORDERKEY
FROM ORDERS
WHERE O_ORDERDATE >= '1994-01-01'
    AND O_ORDERDATE <= '1994-02-01' 
    AND O_TOTALPRICE > (
        SELECT L_EXTENDEDPRICE
        FROM LINEITEM
        WHERE L_ORDERKEY = O_ORDERKEY
);

--Check selectivity of outer table
-- SELECT O_ORDERKEY, O_ORDERPRIORITY, O_ORDERDATE
-- FROM ORDERS
-- WHERE O_ORDERDATE >= '1994-01-01'
--     AND O_ORDERDATE <= '1995-01-01' 
-- );

--Experiment 1 -> Servey a day. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 2 -> Servey a week. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 3 -> Servey a month. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 4 -> Servey a year. Scale factor { 0.5, 1, 2, 4, 8 }


