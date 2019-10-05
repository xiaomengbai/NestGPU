--TPC-H Q16 (Type N)
select
   p_brand,
   p_type,
   p_size,
   count(distinct ps_suppkey) as supplier_cnt
from
   partsupp,
   part,
   supplier 
where
   p_partkey = ps_partkey
   and p_brand <> 'Brand#45'
   and p_type not like 'MEDIUM POLISHED%'
   and s_comment like '%Customer%Complaints%'
   and p_size in (49, 14, 23, 45, 19, 3, 36, 9) 
   and ps_suppkey <> s_suppkey
group by
   p_brand,
   p_type,
   p_size
order by
   supplier_cnt desc,
   p_brand,
   p_type,
   p_size;
