import os
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

load_dotenv(BASE_DIR / ".env")

# =====================================================================
# 1. FUENTE DE DATOS INTERNOS: REGISTRO DE MEDICAMENTOS ISP (IE3)
# =====================================================================
RUTA_CSV = BASE_DIR / "Productos_farmaceuticos_vigentes_venta_directa.csv"
RESPUESTA_SIN_REGISTRO = "No dispongo de registros oficiales en la bodega para responder a esta operación."


def cargar_inventario(ruta_csv: Path) -> pd.DataFrame:
    """Carga el inventario real y permite UTF-8 o Latin-1."""
    columnas = ["N° Registro", "Nombre Producto", "Razon Social Titular", "Condicion Venta"]
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de inventario: {ruta_csv}")

    ultimo_error = None
    for codificacion in ("utf-8", "latin-1"):
        try:
            inventario = pd.read_csv(ruta_csv, sep=";", encoding=codificacion)
            faltantes = [columna for columna in columnas if columna not in inventario.columns]
            if faltantes:
                raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")
            return inventario.fillna("")
        except (UnicodeDecodeError, ValueError) as error:
            ultimo_error = error

    raise ValueError(f"No fue posible leer el inventario: {ultimo_error}")


try:
    df_inventario = cargar_inventario(RUTA_CSV)
    print(f"Éxito: se cargaron {len(df_inventario)} productos farmacéuticos.")
except (FileNotFoundError, ValueError) as error:
    print(f"Error de inventario: {error}")
    df_inventario = pd.DataFrame(
        columns=["N° Registro", "Nombre Producto", "Razon Social Titular", "Condicion Venta"]
    )


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter)).upper()


PALABRAS_IGNORADAS = {
    "PARA", "CUAL", "COMO", "DONDE", "ESTA", "ESTE", "ESTO", "TIENE",
    "SOBRE", "ENTRE", "BODEGA", "PRODUCTO", "MEDICAMENTO", "FARMACEUTICA",
    "FARMACEUTICO", "CUALES", "CUALQUIER", "DESDE", "HASTA", "OPERACION"
}


def buscar_datos_producto(pregunta: str) -> str:
    """Busca un registro por nombre o número de registro sanitario en el inventario oficial."""
    if df_inventario.empty:
        return RESPUESTA_SIN_REGISTRO

    pregunta_normalizada = normalizar(pregunta)
    coincidencias = df_inventario[
        df_inventario.apply(
            lambda fila: (
                normalizar(fila["N° Registro"]) in pregunta_normalizada
                or any(
                    len(palabra) >= 4
                    and palabra not in PALABRAS_IGNORADAS
                    and palabra in pregunta_normalizada
                    for palabra in normalizar(fila["Nombre Producto"]).split()
                )
            ),
            axis=1,
        )
    ]
    if coincidencias.empty:
        return RESPUESTA_SIN_REGISTRO
    return coincidencias.head(5).to_string(index=False)


# =====================================================================
# 2. FUENTE DE DATOS EXTERNOS: MANUAL OPERATIVO EN BASE VECTORIAL (IE3)
# =====================================================================
manual_operativo_bodega = [
    "NORMA TÉCNICA 01 (VACUNAS): Las vacunas y suspensiones inyectables antiinfluenza son altamente sensibles. Exigen almacenamiento estricto en cadena de frío entre 2°C y 8°C. Prohibido congelar.",
    "NORMA TÉCNICA 02 (TÓPICOS): Los ungüentos, cremas y geles tópicos deben mantenerse en estanterías secas a temperatura ambiente controlada, nunca superior a los 25°C. Evitar la exposición directa al sol.",
    "NORMA TÉCNICA 03 (DESPACHO): Todo producto farmacéutico bajo 'Condición de Venta Directa' debe ser verificado visualmente en su rotulación de precios y protegido de la humedad antes de ser cargado al camión de distribución."
]

vectorstore = None


def obtener_retriever_manual():
    """Crea FAISS con embeddings locales solo cuando se realiza una consulta válida."""
    global vectorstore
    if vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(manual_operativo_bodega, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# =====================================================================
# 3. DISEÑO DE PROMPT Y CONFIGURACIÓN DEL MODELO (IE2)
# =====================================================================
llm = None


def obtener_llm():
    """Instancia el modelo Gemini 3.8 Flash con temperature=0."""
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    return llm


template_sistema = ChatPromptTemplate.from_messages([
    ("system", (
        "Eres el Asistente Experto en Logística Farmacéutica de la bodega de MedisChil S.A.\n"
        "Tu tarea es responder consultas operativas del personal combinando los datos del producto "
        "y las normas técnicas del manual de la bodega.\n\n"
        "SÉ CONCISO Y UTILIZA EXCLUSIVAMENTE ESTE CONTEXTO OFICIAL:\n"
        "-----------------------------------------\n"
        "REGISTRO DEL PRODUCTO (DATOS INTERNOS):\n{contexto_inventario}\n\n"
        "MANUAL OPERATIVO DE BODEGA (DATOS EXTERNOS):\n{contexto_manual}\n"
        "-----------------------------------------\n\n"
        "RESTRICCIÓN: Si la pregunta no se puede responder con el contexto anterior, di estrictamente: "
        "'No dispongo de registros oficiales en la bodega para responder a esta operación.'"
    )),
    ("human", "{pregunta}")
])


# =====================================================================
# 4. ORQUESTACIÓN DE LA ARQUITECTURA LCEL (IE4)
# =====================================================================
def ejecutar_consulta_logistica(pregunta_usuario: str):
    # Recuperación de datos estructurados (Inventario)
    contexto_inv = buscar_datos_producto(pregunta_usuario)

    # La fuente oficial decide si la consulta puede llegar al modelo (Guardrail / Short-circuit)
    if contexto_inv == RESPUESTA_SIN_REGISTRO:
        return RESPUESTA_SIN_REGISTRO

    # Recuperación de datos no estructurados (Manual Vectorial con FAISS)
    retriever_manual = obtener_retriever_manual()
    docs_manual = retriever_manual.invoke(pregunta_usuario)
    contexto_man = (
        "\n\n".join(doc.page_content for doc in docs_manual)
        if docs_manual
        else "No se encontraron normas aplicables."
    )

    # Construcción de la cadena ejecutable moderna (LCEL)
    cadena_rag = (
        {
            "contexto_inventario": lambda x: contexto_inv,
            "contexto_manual": lambda x: contexto_man,
            "pregunta": RunnablePassthrough()
        }
        | template_sistema
        | obtener_llm()
        | StrOutputParser()
    )

    return cadena_rag.invoke(pregunta_usuario)


# =====================================================================
# 5. PRUEBA DE COHERENCIA Y EVALUACIÓN (IE6)
# =====================================================================
if __name__ == "__main__":
    consultas_ejemplo = [
        "¿Cuál es el protocolo de almacenamiento y la condición de venta para la vacuna INFLUVAC?",
        "¿Cuáles son los requisitos de almacenamiento y despacho para el producto inexistente FANTASMIN 500?",
        "¿Cómo debe almacenarse y despacharse el ungüento BACITOPIC?"
    ]

    for idx, consulta in enumerate(consultas_ejemplo, 1):
        print(f"==================================================")
        print(f"📦 PRUEBA #{idx} - CONSULTA OPERARIO:")
        print(f"{consulta}\n")
        respuesta_agente = ejecutar_consulta_logistica(consulta)
        print(f"🤖 RESPUESTA SISTEMA:\n{respuesta_agente}\n")
