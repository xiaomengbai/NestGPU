-- Query (Type J - Un-nested): Find all orders within a month that had more than one item.
-- With this query we control the size of the outer table.

SELECT O_ORDERKEY
FROM LINEITEM, ORDERS
WHERE L_ORDERKEY = O_ORDERKEY
    AND  O_TOTALPRICE > L_EXTENDEDPRICE
    AND O_ORDERDATE > '1994-01-01'
    AND O_ORDERDATE < '1994-01-10'
GROUP BY O_ORDERKEY;

--Experiment 1 -> Servey a day. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 2 -> Servey a week. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 3 -> Servey a month. Scale factor { 0.5, 1, 2, 4, 8 }
--Experiment 4 -> Servey a year. Scale factor { 0.5, 1, 2, 4, 8 }