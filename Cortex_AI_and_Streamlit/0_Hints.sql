-- Conéctate a Snowflake y revisa
-- Validación del database y esquema
LIST @CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.RAW_DATA;


--La streamlits apps deben ser creadas con el cortex_user_role

--Problema al usar el LLM con Cortex Analyst, se soluciona cambiando de región:
--use role accountadmin; 

ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'AWS_US';
