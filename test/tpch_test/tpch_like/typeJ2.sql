-- Query (Type J2): Get all the customers from FRANCE that have urgent orders
-- No selectivity. Try to get correct query plan!

SELECT C_NAME
FROM CUSTOMER, NATION
WHERE C_NATIONKEY = N_REGIONKEY
    AND N_NAME = 'FRANCE'
    AND '1-URGENT' in (
        SELECT O_ORDERPRIORITY
        FROM ORDERS
        WHERE O_CUSTKEY = C_CUSTKEY
);