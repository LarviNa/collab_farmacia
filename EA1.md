# Solución RA1: Asistente de logística farmacéutica

## 1. Propósito

La solución implementa un asistente RAG para apoyar operaciones de almacenamiento y despacho en una bodega farmacéutica. Combina:

- Un inventario oficial de productos en formato CSV (2.428 registros del ISP).
- Un manual operativo con normas técnicas de almacenamiento y despacho.
- FAISS con embeddings locales (`all-MiniLM-L6-v2`) para recuperar semánticamente la norma más relacionada.
- `gemini-3.8-flash` de Google para redactar una respuesta operacional basada solo en el contexto recuperado con `temperature=0`.

El alcance corresponde a la RA1: fundamentos de IA generativa, prompt engineering, recuperación aumentada y control del contexto.

## 2. Flujo de funcionamiento

1. `Solucion.py` carga la variable `GOOGLE_API_KEY` desde el entorno o archivo `.env`.
2. Pandas lee `Productos_farmaceuticos_vigentes_venta_directa.csv` usando `;` como separador.
3. Se validan las columnas oficiales y se acepta UTF-8 o Latin-1.
4. La pregunta se normaliza y se busca un producto por nombre o número de registro sanitario.
5. Si no existe un producto oficial, el sistema devuelve exactamente:

   > No dispongo de registros oficiales en la bodega para responder a esta operación.

6. Si existe, FAISS recupera la norma operativa más relacionada con la consulta ($k=2$).
7. LCEL combina el inventario, la norma y la pregunta mediante el operador `|`.
8. `ChatGoogleGenerativeAI` genera una respuesta determinista con `temperature=0`.
9. `StrOutputParser` convierte la salida del modelo en texto estructurado.

## 3. Normas incluidas

- Vacunas y suspensiones: conservar entre 2 °C y 8 °C; no congelar.
- Cremas, geles y ungüentos: conservar en un lugar seco, a una temperatura máxima de 25 °C y sin exposición directa al sol.
- Despacho de venta directa: revisar la rotulación de precios y proteger el producto de la humedad.

## 4. Componentes RAG

En esta solución cada norma operativa breve se conserva como un chunk independiente. No se utiliza un divisor automático porque las tres fuentes ya son textos cortos y completos; dividirlas más podría separar una regla de su condición o excepción.

- **Chunks:** tres normas operativas, una por documento.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (alta velocidad, sin consumo de API).
- **Vectorstore:** FAISS en memoria.
- **LLM:** `gemini-3.8-flash` (Google GenAI).
