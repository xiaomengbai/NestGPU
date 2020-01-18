-- Query (Type J2 Un-nested): Get all the customers from FRANCE that have urgent orders
-- No selectivity. Try to get correct query plan!
-- Un-nested version

SELECT C_NAME
FROM CUSTOMER, NATION, ORDERS
WHERE C_NATIONKEY = N_REGIONKEY
    AND  O_CUSTKEY = C_CUSTKEY
    AND N_NAME = 'FRANCE'
    AND O_ORDERPRIORITY = '1-URGENT'  
GROUP BY C_NAME
;