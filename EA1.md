# Solución RA1: Asistente de logística farmacéutica

## 1. Propósito

La solución implementa un asistente RAG para apoyar operaciones de almacenamiento y despacho en una bodega farmacéutica. Combina:

- Un inventario oficial de productos en formato CSV.
- Un manual operativo con normas de almacenamiento.
- FAISS para recuperar semánticamente la norma más relacionada.
- `gpt-4o-mini` para redactar una respuesta operacional basada solo en el contexto recuperado.

El alcance corresponde a la RA1: fundamentos de IA generativa, prompt engineering, recuperación aumentada y control del contexto.

## 2. Flujo de funcionamiento

1. `Solucion.py` carga la variable `OPENAI_API_KEY` desde `.env`.
2. Pandas lee `Productos_farmaceuticos_vigentes_venta_directa.csv` usando `;` como separador.
3. Se validan las columnas oficiales y se acepta UTF-8 o Latin-1.
4. La pregunta se normaliza y se busca un producto por nombre o número de registro sanitario.
5. Si no existe un producto oficial, el sistema devuelve exactamente:

   > No dispongo de registros oficiales en la bodega para responder a esta operación.

6. Si existe, FAISS recupera la norma operativa más relacionada con la consulta.
7. LCEL combina el inventario, la norma y la pregunta mediante el operador `|`.
8. `ChatOpenAI` genera una respuesta determinista porque usa `temperature=0`.
9. `StrOutputParser` convierte la salida del modelo en texto.

## 3. Normas incluidas

- Vacunas y suspensiones: conservar entre 2 °C y 8 °C; no congelar.
- Cremas, geles y ungüentos: conservar en un lugar seco, a una temperatura máxima de 25 °C y sin exposición directa al sol.
- Despacho de venta directa: revisar la rotulación de precios y proteger el producto de la humedad.

## 4. Componentes RAG

En esta solución cada norma operativa breve se conserva como un chunk independiente. No se utiliza un divisor automático porque las tres fuentes ya son textos cortos y completos; dividirlas más podría separar una regla de su condición o excepción.

- **Chunks:** tres normas operativas, una por documento.
- **Embeddings:** `text-embedding-3-small` transforma cada norma y la consulta en vectores.
- **Índice vectorial:** FAISS mantiene los vectores localmente durante la ejecución.
- **Recuperador:** recupera las dos normas más relevantes (`k=2`) para cubrir consultas que combinen almacenamiento y despacho.
- **Fuente interna:** Pandas busca el producto en el CSV y aporta sus datos oficiales.
- **Generación:** el prompt reúne ambos contextos y `gpt-4o-mini` redacta la respuesta.

Esta separación permite justificar que la solución integra una fuente interna estructurada y una fuente externa no estructurada, como exige el flujo RAG.

## 5. Configuración

Se recomienda Python 3.14, que es el intérprete usado para validar esta solución.

1. Copia `.env.example` y renómbralo como `.env`.
2. Reemplaza `tu_clave_de_openai` por tu clave real.
3. Instala las dependencias:

```powershell
python -m pip install -r Requisitos.txt
```

4. Ejecuta el programa:

```powershell
python Solucion.py
```

Nunca publiques `.env` ni escribas la clave directamente en `Solucion.py`. `.gitignore` ya está configurado para excluir ese archivo.

## 6. Relación con la descripción del proyecto

La descripción indica un sistema que procesa manuales operativos y reportes de inventario para apoyar decisiones en bodega. La implementación actual cumple estas partes:

- **Manuales operativos:** se representan como documentos y se indexan en FAISS.
- **Inventario:** se procesa desde el CSV real mediante Pandas.
- **Apoyo a decisiones:** responde protocolos de almacenamiento y despacho según el producto consultado.
- **Control de información:** rechaza consultas de productos que no existen en el registro oficial.

Actualmente el inventario es un CSV y no existe un módulo separado para reportes históricos, métricas o recomendaciones automáticas. Por eso, la solución funciona como un asistente de consulta y apoyo operacional, no como un sistema completo de optimización logística.

## 7. Evidencia de validación

Se verificó localmente que:

- El CSV carga 2.428 productos.
- Una búsqueda de `INFLUVAC` encuentra registros.
- Una búsqueda de un producto inexistente activa la respuesta anti-alucinación.
- `Solucion.py` compila sin errores.

La respuesta RAG completa requiere una `OPENAI_API_KEY` válida porque necesita generar embeddings y consultar el modelo.
