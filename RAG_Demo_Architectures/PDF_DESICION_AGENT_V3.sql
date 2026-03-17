-- =====================================================
-- PROYECTO: AGENTE CON ÁRBOL DE DECISIÓN + PDFs + BASE DE DATOS
-- =====================================================
-- Flujo: Usuario pregunta → Árbol de decisión → PDFs → Base de datos → Respuesta
-- =====================================================

-- =====================================================
-- PARTE 0: CREAR BASE DE DATOS Y SCHEMA
-- =====================================================

CREATE OR REPLACE DATABASE PDF_KNOWLEDGE_BASE_V3;
CREATE OR REPLACE SCHEMA PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH;

USE DATABASE PDF_KNOWLEDGE_BASE_V3;
USE SCHEMA DOC_SEARCH;

-- =====================================================
-- PARTE 1: CONFIGURACIÓN DE INFRAESTRUCTURA
-- =====================================================

-- 1.1) Stage para PDFs
CREATE OR REPLACE STAGE PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_DOCUMENTS_STAGE
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- 1.2) Stage para CSV (árbol de decisión)
CREATE OR REPLACE STAGE PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.CSV_DECISION_STAGE
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- 1.3) Stage para modelo semántico (Cortex Analyst)
CREATE OR REPLACE STAGE PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.SEMANTIC_MODEL_STAGE
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- 1.4) Tabla para chunks de PDFs
CREATE OR REPLACE TABLE PDF_CHUNKS (
    doc_id STRING,
    doc_name STRING,
    relative_path STRING,
    page_number INT,
    chunk_id STRING,
    content STRING,
    source_type STRING DEFAULT 'pdf',
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 1.5) Tabla para árbol de decisión
CREATE OR REPLACE TABLE DECISION_TREE (
    pregunta STRING,
    respuesta STRING,
    source_type STRING DEFAULT 'decision_tree',
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- =====================================================
-- PARTE 2: CARGA DE DATOS - PDFs
-- =====================================================

-- 2.1) Verificar PDFs en el stage
SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED
FROM DIRECTORY(@PDF_DOCUMENTS_STAGE);

-- 2.2) Parseo de PDFs
CREATE OR REPLACE TABLE PDF_RAW_PARSE AS
SELECT
    RELATIVE_PATH AS relative_path,
    SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
        @PDF_DOCUMENTS_STAGE, 
        RELATIVE_PATH, 
        {'mode': 'LAYOUT'}
    ) AS parsed
FROM DIRECTORY(@PDF_DOCUMENTS_STAGE);

-- 2.3) Insertar chunks parseados
INSERT INTO PDF_CHUNKS (doc_id, doc_name, relative_path, page_number, chunk_id, content, source_type)
SELECT
    SHA2(relative_path, 256) AS doc_id,
    SPLIT_PART(relative_path, '/', -1) AS doc_name,
    relative_path,
    NULL AS page_number,
    CONCAT(SHA2(relative_path, 256), '_full') AS chunk_id,
    TO_VARCHAR(parsed:content) AS content,
    'pdf' AS source_type
FROM PDF_RAW_PARSE;

SELECT * FROM PDF_CHUNKS
SELECT * FROM PDF_RAW_PARSE

-- =====================================================
-- PARTE 3: CARGA DE DATOS - ÁRBOL DE DECISIÓN
-- =====================================================

-- 3.1) Verificar CSV en el stage
SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED
FROM DIRECTORY(@CSV_DECISION_STAGE);

-- 3.2) Cargar CSV (guardar archivo como UTF-8 antes de subir)
COPY INTO DECISION_TREE (pregunta, respuesta)
FROM @CSV_DECISION_STAGE/ARBOL_DESICION.csv
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_DELIMITER = ',' FIELD_OPTIONALLY_ENCLOSED_BY = '"' ENCODING = 'UTF8');

-- =====================================================
-- PARTE 4: CONFIGURAR BASE DE DATOS PARA CORTEX ANALYST
-- =====================================================
-- NOTA: Debes crear un modelo semántico YAML que describa tu base de datos
-- y subirlo al stage SEMANTIC_MODEL_STAGE

--4.1) Crear base de datos

-- 4.2) Verificar modelo semántico en el stage
SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED
FROM DIRECTORY(@SEMANTIC_MODEL_STAGE);

-- =====================================================
-- PARTE 5: CREAR CORTEX SEARCH SERVICES
-- =====================================================

-- 5.1) Cortex Search para PDFs
CREATE OR REPLACE CORTEX SEARCH SERVICE PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_SEARCH_SERVICE
    ON CONTENT
    ATTRIBUTES DOC_NAME, RELATIVE_PATH, PAGE_NUMBER, SOURCE_TYPE
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 hour'
    AS (
        SELECT 
            CONTENT,
            DOC_NAME,
            RELATIVE_PATH,
            PAGE_NUMBER,
            SOURCE_TYPE
        FROM PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_CHUNKS
    );

-- 5.2) Cortex Search para árbol de decisión
CREATE OR REPLACE CORTEX SEARCH SERVICE PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.DECISION_TREE_SERVICE
    ON respuesta
    ATTRIBUTES pregunta, source_type
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 hour'
    AS (
        SELECT 
            pregunta,
            respuesta,
            source_type
        FROM PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.DECISION_TREE
    );

-- =====================================================
-- PARTE 6: CREAR AGENTE CON 3 FUENTES DE DATOS
-- =====================================================

CREATE OR REPLACE AGENT PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_DECISION_DB_AGENT
    COMMENT = 'Agente que consulta: 1) Árbol de decisión, 2) PDFs, 3) Base de datos'
    FROM SPECIFICATION $$
    {
        "models": {
            "orchestration": "claude-4-sonnet"
        },
        "instructions": {
            "orchestration": "FLUJO OBLIGATORIO EN ORDEN: 1) PRIMERO consulta el árbol de decisión (decision_tree_search) para obtener contexto inicial y guía sobre cómo responder. 2) SEGUNDO usa la información del árbol para buscar detalles en los PDFs (pdf_search). 3) TERCERO, si el árbol de decisión o los PDFs indican que se necesitan datos específicos, métricas o información estructurada, consulta la base de datos (database_analyst). Combina las tres fuentes para dar una respuesta completa.",
            "response": "Responde en español. Sé claro y conciso. Indica de qué fuente proviene cada parte de la información (árbol de decisión, documento PDF o base de datos)."
        },
        "tools": [
            {
                "tool_spec": {
                    "type": "cortex_search",
                    "name": "decision_tree_search",
                    "description": "PASO 1 - USAR PRIMERO. Busca en el árbol de decisión para obtener contexto inicial y guía sobre cómo responder la pregunta del usuario."
                }
            },
            {
                "tool_spec": {
                    "type": "cortex_search",
                    "name": "pdf_search",
                    "description": "PASO 2 - USAR SEGUNDO. Busca información detallada en los documentos PDF después de consultar el árbol de decisión."
                }
            },
            {
                "tool_spec": {
                    "type": "cortex_analyst_text_to_sql",
                    "name": "database_analyst",
                    "description": "PASO 3 - USAR AL FINAL SI ES NECESARIO. Consulta la base de datos para obtener datos específicos, métricas, estadísticas o información estructurada cuando el árbol de decisión o los PDFs lo indiquen."
                }
            }
        ],
        "tool_resources": {
            "decision_tree_search": {
                "search_service": "PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.DECISION_TREE_SERVICE",
                "max_results": 5
            },
            "pdf_search": {
                "search_service": "PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_SEARCH_SERVICE",
                "max_results": 10
            },
            "database_analyst": {
                "semantic_model_file": "@PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.SEMANTIC_MODEL_STAGE/DATABASES_CONTEXT.yaml",
                "execution_environment": {
                    "type": "warehouse",
                    "warehouse": "COMPUTE_WH"
                }
            }
        }
    }
    $$;

-- 6.1) Otorgar permisos
GRANT USAGE ON AGENT PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.PDF_DECISION_DB_AGENT TO ROLE PUBLIC;

-- =====================================================
-- PARTE 7: VERIFICACIONES
-- =====================================================

-- 7.1) Verificar PDFs cargados
SELECT COUNT(*) AS n_chunks, COUNT(DISTINCT doc_id) AS n_docs FROM PDF_CHUNKS;

-- 7.2) Resumen por documento
SELECT doc_name, COUNT(*) AS chunks
FROM PDF_CHUNKS
GROUP BY doc_name
ORDER BY chunks DESC;

-- 7.3) Verificar árbol de decisión
SELECT COUNT(*) AS total_reglas FROM DECISION_TREE;
SELECT * FROM DECISION_TREE LIMIT 10;

-- 7.4) Verificar agente
SHOW AGENTS IN SCHEMA PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH;

---
SHOW AGENTS IN SCHEMA PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH

SELECT CURRENT_ACCOUNT(), CURRENT_REGION()

--Api test
https://abc32220.us-east-1.snowflakecomputing.com/api/v2/databases/PDF_KNOWLEDGE_BASE_V3/schemas/DOC_SEARCH/agents/PDF_DECISION_DB_AGENT:run

-- 1. Ver tu IP actual (desde donde usas Postman)
SELECT CURRENT_CLIENT();

-- 2. Crear una network policy que permita tu IP
CREATE OR REPLACE NETWORK POLICY ALLOW_ALL_ACCESS
    ALLOWED_IP_LIST = ('0.0.0.0/0')
    COMMENT = 'Permite todas las IPs - SOLO PARA PRUEBAS';

ALTER ACCOUNT SET NETWORK_POLICY = ALLOW_ALL_ACCESS;

-- 3. Activar la policy a nivel de cuenta
ALTER ACCOUNT SET NETWORK_POLICY = ALLOW_ALL_ACCESS;

CREATE OR REPLACE NETWORK POLICY POSTMAN_ACCESS
    ALLOWED_IP_LIST = ('123.45.67.89/32');  -- Tu IP aquí

SELECT CURRENT_CLIENT() AS TU_IP_ACTUAL

CREATE OR REPLACE NETWORK POLICY POSTMAN_ACCESS_TEMP
    ALLOWED_IP_LIST = ('0.0.0.0/0')
    COMMENT = 'Permite todas las IPs - TEMPORAL PARA PRUEBAS'

ALTER ACCOUNT SET NETWORK_POLICY = POSTMAN_ACCESS_TEMP

LIST @PDF_KNOWLEDGE_BASE_V3.DOC_SEARCH.SEMANTIC_MODEL_STAGE

--

SELECT * 
FROM SNOWFLAKE.ACCOUNT_USAGE.EXTERNAL_ACCESS_HISTORY
ORDER BY START_TIME DESC;

