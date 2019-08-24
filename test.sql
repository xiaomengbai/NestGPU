-- q1_1
-- select sum(lo_extendedprice*lo_discount) as revenue
-- from lineorder,ddate
-- where lo_orderdate = d_datekey
-- and d_year = 1993 and lo_discount>=1
-- and lo_discount<=3
-- and lo_quantity<25;

-- q1_2
-- select sum(lo_extendedprice*lo_discount) as revenue
-- from lineorder,ddate
-- where lo_orderdate = d_datekey
-- and d_yearmonthnum = 199401
-- and lo_discount>=4
-- and lo_discount<=6
-- and lo_quantity>=26
-- and lo_quantity<=35;

-- q1_3
-- select sum(lo_extendedprice*lo_discount) as revenue
-- from lineorder,ddate
-- where lo_orderdate = d_datekey
-- and d_weeknuminyear = 6
-- and d_year = 1994
-- and lo_discount>=5
-- and lo_discount<=7
-- and lo_quantity>=26
-- and lo_quantity<=35;

-- q2_1
-- select sum(lo_revenue),d_year,p_brand1
-- from lineorder,part,supplier,ddate
-- where lo_orderdate = d_datekey
-- and lo_partkey = p_partkey
-- and lo_suppkey = s_suppkey
-- and p_category = 'MFGR#12'
-- and s_region = 'AMERICA'
-- group by d_year,p_brand1
-- order by d_year,p_brand1;

-- q2_2
-- select sum(lo_revenue),d_year,p_brand1
-- from lineorder, part, supplier,ddate
-- where lo_orderdate = d_datekey
-- and lo_partkey = p_partkey
-- and lo_suppkey = s_suppkey
-- and p_brand1 >= 'MFGR#2221'
-- and p_brand1 <= 'MFGR#2228'
-- and s_region = 'ASIA'
-- group by d_year,p_brand1
-- order by d_year,p_brand1;

-- q2_3
-- select sum(lo_revenue),d_year,p_brand1
-- from lineorder,part,supplier,ddate
-- where lo_orderdate = d_datekey
-- and lo_partkey = p_partkey
-- and lo_suppkey = s_suppkey
-- and p_brand1 = 'MFGR#2239'
-- and s_region = 'EUROPE'
-- group by d_year,p_brand1
-- order by d_year,p_brand1;

-- q3_1
-- select c_nation,s_nation,d_year,sum(lo_revenue) as revenue
-- from lineorder,customer, supplier,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_orderdate = d_datekey
-- and c_region = 'ASIA'
-- and s_region = 'ASIA'
-- and d_year >=1992 and d_year <= 1997
-- group by c_nation,s_nation,d_year
-- order by d_year asc,revenue desc;

-- q3_2
-- select c_city,s_city,d_year,sum(lo_revenue) as revenue
-- from lineorder,customer,supplier,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_orderdate = d_datekey
-- and c_nation = 'UNITED STATES'
-- and s_nation = 'UNITED STATES'
-- and d_year >=1992 and d_year <= 1997
-- group by c_city,s_city,d_year
-- order by d_year asc,revenue desc;

-- q3_3
-- select c_city,s_city,d_year,sum(lo_revenue) as revenue
-- from lineorder,customer,supplier,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_orderdate = d_datekey
-- and (c_city = 'UNITED KI1' or c_city = 'UNITED KI5')
-- and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5')
-- and d_year >=1992 and d_year <= 1997
-- group by c_city,s_city,d_year
-- order by d_year asc,revenue desc;

-- q3_4
-- select c_city,s_city,d_year,sum(lo_revenue) as revenue
-- from lineorder,customer,supplier,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_orderdate = d_datekey
-- and (c_city = 'UNITED KI1' or c_city = 'UNITED KI5')
-- and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5')
-- and d_yearmonth = 'Dec1997'
-- group by c_city,s_city,d_year
-- order by d_year asc,revenue desc;

-- select c_city,s_city,d_year
-- from customer,supplier,ddate
-- where -- (c_city = 'UNITED KI1' or c_city = 'UNITED KI5')
-- -- and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5')
-- -- and
-- c_custkey > 10
-- and c_custkey < 20
-- and c_custkey > (select min(s_suppkey) from supplier where s_region = 'AMERICA' and c_city = 'UNITED KT1')
-- and c_custkey < (select max(s_suppkey) from supplier where s_region = 'AMERICA' and c_city = 'UNITED KT1')
-- ;

-- select from two tables
-- select avg(c_custkey) from customer where c_nation = 'MOROCCO';
-- select avg(s_suppkey) from supplier where s_nation = 'MOROCCO';

-- The subquery example
-- select c_name, c_nation, c_city
-- from customer
-- where  c_custkey < (select avg(s_suppkey) from supplier where c_nation = s_nation)
-- and c_custkey > 3;

-- select c_name, c_nation
-- from customer
-- where c_nation in ('MOROCCO', 'JORDAN', 'ARGENTINA');

-- select c_name, c_nation
-- from customer
-- where c_nation like '%O%N';
-- like '%MO%CO%';

select min(c_custkey), c_nation
from customer
group by c_nation;

-- select c_name, c_nation
-- from customer
-- where c_nation in (select s_nation from supplier where s_suppkey > 98);


-- select c_name, c_nation, c_city
-- from customer
-- where  c_custkey < (select avg(s_suppkey) from supplier where c_nation = s_nation and s_suppkey < (select avg(p_partkey) from part where p_size < 20)));

-- select c_name, c_city, s_name, s_city
-- from customer, supplier
-- where c_city = s_city and c_nation = 'MOROCCO';

-- select c_name, s_name
-- from customer, supplier
-- where s_suppkey > (select min(p_partkey) from part where s_city = p_color and s_region = p_color)
-- and c_nation = 'BRAZIL'
-- and s_nation = 'BRAZIL'
-- and c_nation = s_nation;

--select c_name, c_nation, c_city
-- s_name, s_nation, s_city--s_nation, s_suppkey
--from customer--,
-- supplier
-- where c_custkey > s_suppkey
--where  c_custkey < (select avg(s_suppkey) from supplier where c_nation = s_nation) -- and s_city = p_color)
-- and s_suppkey > (select min(p_partkey) from part where s_city = p_color and s_region = p_color)
-- and
-- s_nation = 'BRAZIL'
-- and
-- c_nation = 'BRAZIL'
-- and s_city = c_city
-- c_nation = s_nation
--;

-- A more complex example
-- select c_name, s_name,  lo_revenue
-- from customer, supplier, lineorder
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and c_custkey > 10
-- and s_city = 'UNITED KT1'
-- and s_city = c_city
-- ;

-- where c_custkey > (select max(s_suppkey) from supplier where s_region = 'AMERICA')
-- having c_custkey > avg(c_custkey) -- does not work


-- q4_1
-- select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit
-- from lineorder,supplier,customer,part, ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_partkey = p_partkey
-- and lo_orderdate = d_datekey
-- and c_region = 'AMERICA'
-- and s_region = 'AMERICA'
-- and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
-- group by d_year,c_nation
-- order by d_year,c_nation;

-- q4_2
-- select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit
-- from lineorder,customer,supplier,part,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_partkey = p_partkey
-- and lo_orderdate = d_datekey
-- and c_region = 'AMERICA'
-- and s_region = 'AMERICA'
-- and (d_year = 1997 or d_year = 1998)
-- and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
-- group by d_year,s_nation, p_category
-- order by d_year,s_nation, p_category;


-- q4_3
-- select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit
-- from lineorder,supplier,customer,part,ddate
-- where lo_custkey = c_custkey
-- and lo_suppkey = s_suppkey
-- and lo_partkey = p_partkey
-- and lo_orderdate = d_datekey
-- and c_region = 'AMERICA'
-- and s_nation = 'UNITED STATES'
-- and (d_year = 1997 or d_year = 1998)
-- and p_category = 'MFGR#14'
-- group by d_year,s_city,p_brand1
-- order by d_year,s_city,p_brand1;



-- select c_name
-- from customer--,supplier
-- where c_custkey > 100
-- and c_region = 'AMERICA'
-- --and c_region = s_region
-- ;

-- select c_name
-- from customer
-- where c_custkey > (select max(s_suppkey) from supplier where s_region = 'AMERICA');


-- select count(*)
-- from customer
-- where c_city = 'BRAZIL';

-- TYPE-N
-- select c_name
-- from customer
-- where c_city in (select s_city from supplier where s_region = 'AMERICA');

-- TYPE-J
-- select c_name
-- from customer
-- where c_city in (select s_city from supplier where c_nation = s_nation);


-- select avg(lo_discount)
-- from lineorder
-- where lo_discount <= 3
-- group by lo_quantity;

-- SUPPLIER|S_SUPPKEY:INTEGER|S_NAME:TEXT:25|S_ADDRESS:TEXT:25|S_CITY:TEXT:10|S_NATION:TEXT:15|S_REGION:TEXT:12|S_PHONE:TEXT:15
-- CUSTOMER|C_CUSTKEY:INTEGER|C_NAME:TEXT:25|C_ADDRESS:TEXT:25|C_CITY:TEXT:10|C_NATION:TEXT:15|C_REGION:TEXT:12|C_PHONE:TEXT:15|C_MKTSEGMENT:TEXT:10


-- select sum(lo_extendedprice*lo_discount) as revenue
-- from lineorder,ddate
-- where lo_orderdate = d_datekey
-- and d_year = 1993 and lo_discount>=1
-- and lo_discount<=3;
-- and lo_quantity<25;
