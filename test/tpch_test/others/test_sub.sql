-- Tesing "SUBQUERY" (Type N) [WORKS]
-- SELECT C_NAME
-- FROM CUSTOMER
-- WHERE C_NATIONKEY = (
--     SELECT N_NATIONKEY FROM NATION WHERE N_NAME = 'MOROCCO');

-- Tesing "SUBQUERY" (Type A) [WORKS]
SELECT S_NAME
FROM SUPPLIER
WHERE S_NATIONKEY = (
    SELECT min (N_NATIONKEY) FROM NATION);
 
    -- Tesing "SUBQUERY" (Type J) [NOT WORKING]
-- SELECT P_NAME 
-- FROM PART 
-- WHERE P_PARTKEY IN (
--         SELECT L_PARTKEY FROM LINEITEM WHERE P_RETAILPRICE = L_EXTENDEDPRICE);

-- Subquery example (Type JA) [WORKS]
-- SELECT
--   s_acctbal,
--   s_name
-- FROM
--   part,
--   partsupp,
--   supplier
-- WHERE
--   p_partkey = ps_partkey
--   and s_suppkey = ps_suppkey
--   and ps_supplycost = (
--     SELECT
--       min(ps_supplycost)
--     FROM
--       partsupp
--     WHERE
--        p_partkey = ps_partkey
--     );