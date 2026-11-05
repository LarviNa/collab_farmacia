# Solución RA1: Asistente de logística farmacéutica

## 1. Propósito

La solución implementa un asistente RAG para apoyar operaciones de almacenamiento y despacho en una bodega farmacéutica. Combina:

- Un inventario oficial de productos en formato CSV (2.428 registros del ISP).
- Un manual operativo con normas técnicas de almacenamiento y despacho.
- FAISS persistente en disco con embeddings locales (`all-MiniLM-L6-v2`) para recuperar semánticamente la norma más relacionada.
- Validación previa en dos fases: búsqueda determinista exacta y **búsqueda difusa (Fuzzy Matching con distancia de Levenshtein)** para tolerar errores tipográficos.
- `gemini-3.8-flash` de Google para redactar una respuesta operacional basada solo en el contexto recuperado con `temperature=0`.

El alcance corresponde a la RA1: fundamentos de IA generativa, prompt engineering, recuperación aumentada y control del contexto.

## 2. Flujo de funcionamiento

1. `Solucion.py` carga la variable `GOOGLE_API_KEY` desde el entorno o archivo `.env`.
2. Pandas lee `Productos_farmaceuticos_vigentes_venta_directa.csv` usando `;` como separador.
3. Se validan las columnas oficiales y se acepta UTF-8 o Latin-1.
4. La pregunta se normaliza y se busca el producto en dos fases:
   - **Fase 1 (Exacta):** Coincidencia directa por número de registro o palabras clave.
   - **Fase 2 (Difusa):** Si no hay coincidencia exacta, se evalúa similitud de Levenshtein ($\ge 0.82$) para corregir erratas como `INFLUVAQ` $\rightarrow$ `INFLUVAC`.
5. Si no existe un producto oficial ni coincidencia difusa, el sistema activa el guardrail/short-circuit y devuelve exactamente:

   > No dispongo de registros oficiales en la bodega para responder a esta operación.

6. Si existe, FAISS recupera la norma operativa más relacionada con la consulta ($k=2$).
7. LCEL combina el inventario, la norma y la pregunta mediante el operador `|`.
8. `ChatGoogleGenerativeAI` genera una respuesta determinista con `temperature=0`.
9. `StrOutputParser` convierte la salida del modelo en texto estructurado.
10. `evaluacion_rag.py` permite evaluar cuantitativamente la fidelidad contextual (*faithfulness*), relevancia y precisión del guardrail.

## 3. Normas incluidas

- Vacunas y suspensiones: conservar entre 2 °C y 8 °C; no congelar.
- Cremas, geles y ungüentos: conservar en un lugar seco, a una temperatura máxima de 25 °C y sin exposición directa al sol.
- Despacho de venta directa: revisar la rotulación de precios y proteger el producto de la humedad.

## 4. Componentes RAG y Persistencia

- **Chunks:** Tres normas operativas, una por documento atómico.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (alta velocidad, sin consumo de API).
- **Vectorstore:** FAISS con persistencia local en disco (`faiss_index_bodega`) y arquitectura desacoplada para migración a motores distribuidos (Qdrant / Pgvector).
- **LLM:** `gemini-3.8-flash` (Google GenAI con `temperature=0.0`).
- **Evaluación RAG:** Framework automatizado con métricas de fidelidad y relevancia contextual.
