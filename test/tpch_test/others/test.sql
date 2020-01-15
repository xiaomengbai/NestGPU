-- ======= WORKING QUERIES =======

-- Test SUPPLIER table [WORKS]
select * from supplier;

-- Test PART table [WORKS]
-- select * from part;

-- Test CUSTOMER table [WORKS]
-- select * from customer;

-- Test NATION table [WORKS]
-- select * from nation;

-- Test REGION table [WORKS]
-- select * from region;

-- Test PARTSUPP table [WORKS]
-- select * from partsupp;

-- Test ORDERS table [WORKS]
-- select * from orders;

-- Test LINEITEM table [NOT WORKING]
-- select * from lineitem;

-- Testing "FILTER" INT [WORKS]
-- select *
-- from customer
-- where C_CUSTKEY = 74991;

-- Testing "FILTER" TEXT [WORKS]
-- select *
-- from customer
-- where C_NAME = 'Customer#000075000';

-- Testing ">" [WORKS]
-- select *
-- from customer
-- where C_CUSTKEY > 74990;

-- Testing "JOIN" [WORKS]
-- select  P_NAME
-- from PART, PARTSUPP
-- where P_PARTKEY = PS_PARTKEY;

-- Testing 3-way "JOIN" [WORKS]
-- select  P_NAME, S_NAME
-- from PART, PARTSUPP, SUPPLIER
-- where P_PARTKEY = PS_PARTKEY
--     and PS_SUPPKEY = S_SUPPKEY;

--Testing GROUP BY [WORKS]
-- select P_BRAND, min(P_RETAILPRICE)
-- from PART
-- group by P_BRAND;

-- Testing "TEXT COMPARISON" [WORKS]
-- select C_NAME, C_COMMENT
-- from CUSTOMER
-- where C_COMMENT like 'fin%';

-- ======= NOT WORKING QUERIES =======

-- Testing order by [NOT WORKING -> Assertion `odNode->table->tupleNum < 1024']
-- select c_custkey
-- from customer
-- order by c_custkey;

-- Testing "COUNT" and ">" [NOT WORKING]
-- select COUNT(*)
-- from customer
-- where C_CUSTKEY > 74990;