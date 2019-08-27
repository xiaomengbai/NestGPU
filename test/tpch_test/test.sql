-- SIMPLE TPC-H query
select S_NAME from supplier;


-- Testing "FILTER"
-- select *
-- from customer
-- where c_city = 'BRAZIL';


-- Testing "COUNT"
-- select count(*)
-- from customer
-- where c_city = 'BRAZIL';


-- Testing "JOIN"
-- select c_name, c_city, s_name, s_city
-- from customer, supplier
-- where c_city = s_city and c_nation = 'MOROCCO';


--Testing "MIN"
--select min(c_custkey), c_nation
--from customer
--group by c_nation;


-- Testing "TEXT COMPARISON"
-- select c_name, c_nation
-- from customer
-- where c_nation like '%O%N';
-- like '%MO%CO%';


-- Tesing "SUBQUERY" (Type N)
-- select c_name
-- from customer
-- where c_city in (select s_city from supplier where s_region = 'AMERICA');


-- Tesing "SUBQUERY" (Type J)
--select c_name
--from customer;
--where c_nationkey in (select s_nationkey from supplier where s_suppkey > 98);


-- Tesing "SUBQUERY" (Type A)
-- select c_name
-- from customer
-- where c_custkey > (select max(s_suppkey) from supplier where s_region = 'AMERICA');


-- Subquery example (Type JA)
--select c_name, c_nation, c_city 
--from customer
--where  c_custkey < (select avg(s_suppkey) from supplier where c_nationkey = s_nationkey and s_suppkey < (select avg(p_partkey) from part where p_size < 20)));
