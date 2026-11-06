"""
agente_medischil.py
Solución Oficial Evaluación Parcial N°2 (EP2) - Asignatura: ISY0101
Asistente Agéntico Funcional Autónomo para MedisChil S.A.

Arquitectura basada en los estándares de las Sesiones 8, 9, 10 y 11:
- Framework: LangGraph (StateGraph con bucle cíclico ReAct adaptativo).
- Core Engine: Google Gemini 3.8 Flash con Native Function Calling y binding estricto.
- Esquemas de Contrato: Pydantic v2 (BaseModel y Field con validaciones de rango/tipo).
- Herramientas: Herencia de BaseTool (Consulta, Razonamiento y Escritura Transaccional Segura).
- Memoria de Corto Plazo (STM): Estado episódico con Annotated[list[BaseMessage], add_messages].
- Memoria Semántica (LTM): FAISS persistente en disco para manual técnico y normativas sanitarias.
- Gobierno y Seguridad: Control de Acceso Basado en Roles (RBAC) y Claves de Idempotencia (UUID v4).
- Observabilidad: Trazabilidad estructurada con Event Sourcing y RotatingFileHandler.
"""

import os
import sys
import json
import uuid
import logging
import unicodedata
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Annotated, TypedDict, Type, Dict, Any, List, Optional, Set

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
import pandas as pd
from pydantic import BaseModel, Field

# Componentes oficiales LangChain y LangGraph
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Carga de variables de entorno
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# =====================================================================
# 1. OBSERVABILIDAD Y REGISTRO AUDITABLE (EVENT SOURCING)
# =====================================================================
def configurar_logger_agente() -> logging.Logger:
    """Configura el registro auditable conforme a estándares de gobernanza ISO/DAMA."""
    logger = logging.getLogger("AgenteMedisChil_EP2")
    logger.setLevel(logging.INFO)

    log_path = BASE_DIR / "auditoria_operacional.log"
    formato = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    manejador = RotatingFileHandler(str(log_path), maxBytes=1024 * 1024 * 5, backupCount=3, encoding="utf-8")
    manejador.setFormatter(formato)

    if not logger.handlers:
        logger.addHandler(manejador)
        # Handler opcional para consola
        consola = logging.StreamHandler(sys.stdout)
        consola.setFormatter(formato)
        consola.setLevel(logging.WARNING)
        logger.addHandler(consola)

    return logger


logger_agente = configurar_logger_agente()


# =====================================================================
# 2. FUENTE DE DATOS: CATÁLOGO ISP CON BÚSQUEDA DETERMINISTA Y DIFUSA
# =====================================================================
RUTA_CSV = BASE_DIR / "Productos_farmaceuticos_vigentes_venta_directa.csv"
RESPUESTA_SIN_REGISTRO = "No dispongo de registros oficiales en la bodega para responder a esta operación."


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).upper()


try:
    from rapidfuzz import fuzz
    def similitud_cadenas(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100.0
except ImportError:
    import difflib
    def similitud_cadenas(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


def cargar_inventario_isp(ruta_csv: Path) -> pd.DataFrame:
    columnas = ["N° Registro", "Nombre Producto", "Razon Social Titular", "Condicion Venta"]
    if not ruta_csv.exists():
        logger_agente.error(f"Catálogo CSV no encontrado: {ruta_csv}")
        return pd.DataFrame(columns=columnas)

    for cod in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(ruta_csv, sep=";", encoding=cod)
            if all(col in df.columns for col in columnas):
                logger_agente.info(f"Catálogo cargado: {len(df)} especialidades farmacéuticas.")
                return df.fillna("")
        except Exception:
            continue

    return pd.DataFrame(columns=columnas)


df_inventario = cargar_inventario_isp(RUTA_CSV)

PALABRAS_IGNORADAS = {
    "PARA", "CUAL", "COMO", "DONDE", "ESTA", "ESTE", "ESTO", "TIENE",
    "SOBRE", "ENTRE", "BODEGA", "PRODUCTO", "MEDICAMENTO", "FARMACEUTICA",
    "FARMACEUTICO", "CUALES", "CUALQUIER", "DESDE", "HASTA", "OPERACION"
}


# =====================================================================
# 3. BASE VECTORIAL DE MEMORIA SEMÁNTICA A LARGO PLAZO (LTM - FAISS)
# =====================================================================
manual_operativo_bodega = [
    "NORMA TÉCNICA 01 (VACUNAS Y BIOLÓGICOS): Exigen almacenamiento estricto en cadena de frío entre 2°C y 8°C. Prohibido congelar. Transporte obligatorio en contenedores térmicos certificados con monitoreo de termógrafo continuo. Toda temperatura superior a 8°C exige bloqueo inmediato en cuarentena preventiva.",
    "NORMA TÉCNICA 02 (PRODUCTOS TÓPICOS): Los ungüentos, cremas y geles deben mantenerse en estanterías secas a temperatura ambiente controlada, nunca superior a 25°C. Evitar la exposición directa al sol y humedad excesiva.",
    "NORMA TÉCNICA 03 (DESPACHO Y VENTA DIRECTA): Todo producto con condición de Venta Directa debe ser verificado visualmente en su rotulación oficial de precios y protegido de la humedad antes de su carga al camión de distribución según Decreto Supremo 466."
]

RUTA_FAISS_LOCAL = BASE_DIR / "faiss_index_bodega"
_retriever_instancia = None


def obtener_retriever_normas():
    """Recupera el motor semántico FAISS con persistencia local."""
    global _retriever_instancia
    if _retriever_instancia is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if RUTA_FAISS_LOCAL.exists() and (RUTA_FAISS_LOCAL / "index.faiss").exists():
            try:
                vectorstore = FAISS.load_local(
                    str(RUTA_FAISS_LOCAL), embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                vectorstore = FAISS.from_texts(manual_operativo_bodega, embeddings)
                vectorstore.save_local(str(RUTA_FAISS_LOCAL))
        else:
            vectorstore = FAISS.from_texts(manual_operativo_bodega, embeddings)
            try:
                vectorstore.save_local(str(RUTA_FAISS_LOCAL))
            except Exception:
                pass
        _retriever_instancia = vectorstore.as_retriever(search_kwargs={"k": 2})
    return _retriever_instancia


# =====================================================================
# 4. SUBSISTEMA TRANSACCIONAL CON RBAC E IDEMPOTENCIA (ESCRITURA SEGURA)
# =====================================================================
class ServidorTransaccionalMedisChil:
    """Subsistema transaccional para escrituras de despacho y actas de cuarentena."""

    def __init__(self) -> None:
        self.ruta_despachos = BASE_DIR / "data_registro_despachos.json"
        self.ruta_cuarentena = BASE_DIR / "data_bitacora_cuarentena.json"
        self.claves_idempotencia_procesadas: Set[str] = set()
        self._inicializar_archivos()

    def _inicializar_archivos(self):
        for ruta in (self.ruta_despachos, self.ruta_cuarentena):
            if not ruta.exists():
                with open(ruta, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
            else:
                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            if "clave_idempotencia" in item:
                                self.claves_idempotencia_procesadas.add(item["clave_idempotencia"])
                except Exception:
                    pass

    def autorizar_orden_despacho(
        self,
        token_rol: str,
        nombre_producto: str,
        n_registro: str,
        cantidad: int,
        destino: str,
        temperatura_transporte: float,
        clave_idempotencia: str
    ) -> Dict[str, Any]:
        # Salvaguarda 1: Control de Acceso Basado en Roles (RBAC)
        roles_autorizados = ["OPERARIO_BODEGA", "SUPERVISOR_LOGISTICA", "DIRECTOR_TECNICO"]
        if token_rol not in roles_autorizados:
            logger_agente.warning(f"[RBAC Denegado] Intento de despacho con rol no autorizado: {token_rol}")
            return {
                "status": "ERROR_SEGURIDAD_RBAC",
                "mensaje": f"El rol '{token_rol}' no posee privilegios para autorizar órdenes de despacho."
            }

        # Salvaguarda 2: Clave de Idempotencia
        if clave_idempotencia in self.claves_idempotencia_procesadas:
            logger_agente.warning(f"[Idempotencia] Clave duplicada: {clave_idempotencia}")
            return {
                "status": "ERROR_IDEMPOTENCIA",
                "mensaje": f"La clave de idempotencia '{clave_idempotencia}' ya fue procesada anteriormente."
            }

        # Salvaguarda 3: Registro persistente
        nueva_orden = {
            "id_despacho": f"DSP-{uuid.uuid4().hex[:8].upper()}",
            "clave_idempotencia": clave_idempotencia,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "producto": nombre_producto,
            "n_registro": n_registro,
            "cantidad_unidades": cantidad,
            "destino": destino,
            "temperatura_camion_c": temperatura_transporte,
            "autorizado_por_rol": token_rol,
            "estado": "AUTORIZADO_PARA_CARGA"
        }

        try:
            with open(self.ruta_despachos, "r+", encoding="utf-8") as f:
                ordenes = json.load(f)
                ordenes.append(nueva_orden)
                f.seek(0)
                json.dump(ordenes, f, indent=2, ensure_ascii=False)
            self.claves_idempotencia_procesadas.add(clave_idempotencia)
            logger_agente.info(f"Orden de despacho emitida con éxito: {nueva_orden['id_despacho']}")
            return {"status": "EXITO", "orden": nueva_orden}
        except Exception as e:
            return {"status": "ERROR_ESCRITURA", "detalle": str(e)}

    def emitir_acta_cuarentena(
        self,
        token_rol: str,
        nombre_producto: str,
        n_registro: str,
        motivo_incumplimiento: str,
        temperatura_detectada: float,
        clave_idempotencia: str
    ) -> Dict[str, Any]:
        """Emisión mandatoria de acta de bloqueo sanitario ante condiciones de riesgo."""
        if clave_idempotencia in self.claves_idempotencia_procesadas:
            return {
                "status": "ERROR_IDEMPOTENCIA",
                "mensaje": f"La clave de idempotencia '{clave_idempotencia}' ya fue procesada."
            }

        acta = {
            "id_acta_cuarentena": f"CUAR-{uuid.uuid4().hex[:8].upper()}",
            "clave_idempotencia": clave_idempotencia,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "producto_bloqueado": nombre_producto,
            "n_registro": n_registro,
            "motivo_sanitario": motivo_incumplimiento,
            "temperatura_critica_c": temperatura_detectada,
            "emitida_por_rol": token_rol,
            "accion_preventiva": "Lote retenido preventivamente en cámara de aislamiento. Notificación inmediata al Director Técnico farmacéutico (Decreto Supremo N° 466)."
        }

        try:
            with open(self.ruta_cuarentena, "r+", encoding="utf-8") as f:
                actas = json.load(f)
                actas.append(acta)
                f.seek(0)
                json.dump(actas, f, indent=2, ensure_ascii=False)
            self.claves_idempotencia_procesadas.add(clave_idempotencia)
            logger_agente.warning(f"ACTA DE CUARENTENA EMITIDA: {acta['id_acta_cuarentena']} para {nombre_producto}")
            return {"status": "BLOQUEO_SANITARIO_REGISTRADO", "acta": acta}
        except Exception as e:
            return {"status": "ERROR_ESCRITURA", "detalle": str(e)}


servidor_transaccional = ServidorTransaccionalMedisChil()


# =====================================================================
# 5. ESQUEMAS PYDANTIC V2 Y HERRAMIENTAS PERSONALIZADAS (BASETOOL)
# =====================================================================

# --- Herramienta 1: Consulta Catálogo ISP ---
class ConsultaCatalogoInput(BaseModel):
    termino_busqueda: str = Field(..., description="Nombre comercial, principio activo o N° de registro sanitario del fármaco a buscar.")


class HerramientaConsultaCatalogoISP(BaseTool):
    name: str = "consultar_catalogo_isp"
    description: str = "Consulta estructurada y difusa en el catálogo oficial de 2.428 medicamentos del ISP. Valida existencia oficial, titular y condición de venta."
    args_schema: Type[BaseModel] = ConsultaCatalogoInput

    def _run(self, termino_busqueda: str) -> str:
        if df_inventario.empty:
            return json.dumps({"status": "ERROR", "mensaje": RESPUESTA_SIN_REGISTRO})

        term_norm = normalizar(termino_busqueda)

        # 1. Búsqueda exacta
        coincidencias = df_inventario[
            df_inventario.apply(
                lambda fila: (
                    normalizar(fila["N° Registro"]) in term_norm
                    or any(
                        len(palabra) >= 4 and palabra not in PALABRAS_IGNORADAS and palabra in term_norm
                        for palabra in normalizar(fila["Nombre Producto"]).split()
                    )
                ),
                axis=1
            )
        ]

        # 2. Búsqueda difusa (Levenshtein) si no hubo coincidencia exacta
        if coincidencias.empty:
            tokens_consulta = [p for p in term_norm.split() if len(p) >= 4 and p not in PALABRAS_IGNORADAS]
            if tokens_consulta:
                def coincide_difuso(fila):
                    reg_norm = normalizar(fila["N° Registro"])
                    for t in tokens_consulta:
                        if similitud_cadenas(t, reg_norm) >= 0.82:
                            return True
                    palabras_prod = [p for p in normalizar(fila["Nombre Producto"]).split() if len(p) >= 4 and p not in PALABRAS_IGNORADAS]
                    for t in tokens_consulta:
                        for pp in palabras_prod:
                            if similitud_cadenas(t, pp) >= 0.82:
                                return True
                    return False

                coincidencias = df_inventario[df_inventario.apply(coincide_difuso, axis=1)]

        if coincidencias.empty:
            return json.dumps({
                "status": "PRODUCTO_NO_REGISTRADO",
                "mensaje": RESPUESTA_SIN_REGISTRO
            })

        top_match = coincidencias.iloc[0]
        resultado = {
            "status": "ENCONTRADO",
            "n_registro": str(top_match["N° Registro"]),
            "nombre_producto": str(top_match["Nombre Producto"]),
            "titular": str(top_match["Razon Social Titular"]),
            "condicion_venta": str(top_match["Condicion Venta"])
        }
        return json.dumps(resultado, ensure_ascii=False)


# --- Herramienta 2: Consulta Normativa Manual Bodega (LTM) ---
class ConsultaNormasInput(BaseModel):
    tipo_o_categoria_medicamento: str = Field(..., description="Categoría del producto (ej. 'vacuna', 'ungüento tópico', 'cadena de frío', 'despacho')")


class HerramientaConsultaNormasBodega(BaseTool):
    name: str = "consultar_normas_bodega"
    description: str = "Recupera semánticamente desde la base vectorial FAISS las normas técnicas de manipulación, temperatura y despacho para esa categoría de fármaco."
    args_schema: Type[BaseModel] = ConsultaNormasInput

    def _run(self, tipo_o_categoria_medicamento: str) -> str:
        retriever = obtener_retriever_normas()
        docs = retriever.invoke(tipo_o_categoria_medicamento)
        if not docs:
            return json.dumps({"status": "SIN_NORMAS", "contenido": "No se encontraron directrices específicas."})
        normas = [d.page_content for d in docs]
        return json.dumps({"status": "NORMAS_RECUPERADAS", "normas": normas}, ensure_ascii=False)


# --- Herramienta 3: Razonamiento y Evaluación de Transporte Sanitario ---
class EvaluacionTransporteInput(BaseModel):
    nombre_producto: str = Field(..., description="Nombre del medicamento.")
    temperatura_camion_c: float = Field(..., description="Temperatura medida actualmente en el vehículo o furgón de despacho (°C).")
    tipo_producto: str = Field(..., description="Tipo o clasificación: 'VACUNA_BIOLOGICO', 'TOPICO', o 'ESTANDAR'.")


class HerramientaEvaluacionTransporte(BaseTool):
    name: str = "evaluar_viabilidad_transporte"
    description: str = "Herramienta de razonamiento sanitario que evalúa si la temperatura del vehículo cumple estrictamente con el rango térmico normativo (DS 466)."
    args_schema: Type[BaseModel] = EvaluacionTransporteInput

    def _run(self, nombre_producto: str, temperatura_camion_c: float, tipo_producto: str) -> str:
        tipo = tipo_producto.upper()
        if "VACUNA" in tipo or "BIOLOGICO" in tipo or "INFLUVAC" in nombre_producto.upper():
            # Rango estricto 2°C a 8°C
            if 2.0 <= temperatura_camion_c <= 8.0:
                return json.dumps({
                    "cumple_normativa": True,
                    "rango_exigido": "2°C a 8°C (Cadena de Frío Estricta)",
                    "temperatura_medida": temperatura_camion_c,
                    "conclusion": "Temperatura dentro de rango. Viable para carga."
                })
            else:
                return json.dumps({
                    "cumple_normativa": False,
                    "rango_exigido": "2°C a 8°C",
                    "temperatura_medida": temperatura_camion_c,
                    "desviacion_critica": f"Rotura de cadena de frío detectada: {temperatura_camion_c}°C está fuera del rango seguro (2°C-8°C).",
                    "accion_requerida": "BLOQUEO_SANITARIO_INMEDIATO_CUARENTENA"
                })
        elif "TOPICO" in tipo or "UNGÜENTO" in nombre_producto.upper() or "CREMA" in nombre_producto.upper():
            if temperatura_camion_c <= 25.0:
                return json.dumps({
                    "cumple_normativa": True,
                    "rango_exigido": "Menor o igual a 25°C",
                    "temperatura_medida": temperatura_camion_c,
                    "conclusion": "Temperatura ambiente controlada correcta."
                })
            else:
                return json.dumps({
                    "cumple_normativa": False,
                    "rango_exigido": "Hasta 25°C",
                    "temperatura_medida": temperatura_camion_c,
                    "desviacion_critica": f"Exceso térmico para tópicos: {temperatura_camion_c}°C supera los 25°C permitidos.",
                    "accion_requerida": "BLOQUEO_SANITARIO_INMEDIATO_CUARENTENA"
                })
        else:
            return json.dumps({
                "cumple_normativa": True,
                "temperatura_medida": temperatura_camion_c,
                "conclusion": "Condiciones estándar aceptables."
            })


# --- Herramienta 4: Escritura - Emisión de Despacho Transaccional ---
class EmisionDespachoInput(BaseModel):
    token_rol: str = Field(default="OPERARIO_BODEGA", description="Rol del usuario autenticado (RBAC).")
    nombre_producto: str = Field(..., description="Nombre oficial del medicamento autorizado.")
    n_registro: str = Field(..., description="Número de registro sanitario oficial ISP.")
    cantidad: int = Field(..., gt=0, description="Cantidad de unidades a despachar (entero mayor a 0).")
    destino: str = Field(..., description="Institución o sucursal de destino (ej. Hospital Central, Farmacia Red 1).")
    temperatura_transporte: float = Field(..., description="Temperatura verificada del vehículo (°C).")
    clave_idempotencia: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Clave unívoca UUID v4 para garantizar idempotencia.")


class HerramientaEmisionDespacho(BaseTool):
    name: str = "emitir_orden_despacho"
    description: str = "Herramienta de escritura transaccional segura. Registra oficialmente una orden de despacho autorizada tras validar normativas y stock."
    args_schema: Type[BaseModel] = EmisionDespachoInput

    def _run(self, token_rol: str, nombre_producto: str, n_registro: str, cantidad: int, destino: str, temperatura_transporte: float, clave_idempotencia: str) -> str:
        resultado = servidor_transaccional.autorizar_orden_despacho(
            token_rol=token_rol,
            nombre_producto=nombre_producto,
            n_registro=n_registro,
            cantidad=cantidad,
            destino=destino,
            temperatura_transporte=temperatura_transporte,
            clave_idempotencia=clave_idempotencia
        )
        return json.dumps(resultado, ensure_ascii=False)


# --- Herramienta 5: Escritura - Bloqueo Sanitario y Cuarentena ---
class BloqueoCuarentenaInput(BaseModel):
    token_rol: str = Field(default="OPERARIO_BODEGA", description="Rol del usuario que reporta la anomalía.")
    nombre_producto: str = Field(..., description="Nombre del medicamento que no superó los estándares.")
    n_registro: str = Field(..., description="Número de registro sanitario oficial ISP.")
    motivo_incumplimiento: str = Field(..., description="Descripción técnica de la desviación o riesgo sanitario detectado.")
    temperatura_detectada: float = Field(..., description="Temperatura crítica registrada que originó el rechazo.")
    clave_idempotencia: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Clave unívoca UUID v4 de idempotencia.")


class HerramientaBloqueoCuarentena(BaseTool):
    name: str = "emitir_acta_cuarentena"
    description: str = "Herramienta de escritura de contingencia. Bloquea de inmediato el lote y emite un acta oficial de cuarentena preventiva sanitaria."
    args_schema: Type[BaseModel] = BloqueoCuarentenaInput

    def _run(self, token_rol: str, nombre_producto: str, n_registro: str, motivo_incumplimiento: str, temperatura_detectada: float, clave_idempotencia: str) -> str:
        resultado = servidor_transaccional.emitir_acta_cuarentena(
            token_rol=token_rol,
            nombre_producto=nombre_producto,
            n_registro=n_registro,
            motivo_incumplimiento=motivo_incumplimiento,
            temperatura_detectada=temperatura_detectada,
            clave_idempotencia=clave_idempotencia
        )
        return json.dumps(resultado, ensure_ascii=False)


# Lista maestra de herramientas expuestas al modelo
catalogo_herramientas = [
    HerramientaConsultaCatalogoISP(),
    HerramientaConsultaNormasBodega(),
    HerramientaEvaluacionTransporte(),
    HerramientaEmisionDespacho(),
    HerramientaBloqueoCuarentena()
]
mapa_herramientas: Dict[str, BaseTool] = {t.name: t for t in catalogo_herramientas}


# =====================================================================
# 6. ESTADO COMPARTIDO (SHARED STATE - STM) CON LANGGRAPH
# =====================================================================
class EstadoAgenteBodega(TypedDict):
    """Estado compartido episódico y transaccional del agente de MedisChil S.A."""
    messages: Annotated[List[BaseMessage], add_messages]
    id_sesion: str
    rol_usuario: str
    trazabilidad_eventos: List[Dict[str, Any]]


# =====================================================================
# 7. MOTOR DE RAZONAMIENTO Y ORQUESTACIÓN (LANGGRAPH)
# =====================================================================
def instanciar_llm_agente():
    """Instancia el motor LLM con Tool Binding determinista (soporta Groq gpt-oss-120b o Gemini 3.8 Flash)."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip().strip("'").strip('"')
    if groq_key:
        modelo_groq = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        llm = ChatGroq(
            model=modelo_groq,
            api_key=groq_key,
            temperature=0.0
        )
        return llm.bind_tools(catalogo_herramientas)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip().strip("'").strip('"')
    modelo = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    llm = ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=api_key if api_key else None,
        temperature=0.0
    )
    return llm.bind_tools(catalogo_herramientas)


PROMPT_SISTEMA_MEDISCHIL = SystemMessage(
    content=(
        "Eres el Agente Autónomo de Soporte y Seguridad Operacional de la bodega de MedisChil S.A. "
        "Tu misión es asistir al personal en consultas de catálogo, aplicación de normativas sanitarias "
        "y ejecución controlada de operaciones de despacho y cuarentena preventiva (DS N° 466 del ISP).\n\n"
        "REGLAS OPERATIVAS ESTRICTAS:\n"
        "1. PRIORIDAD SANITARIA: La seguridad de los medicamentos es primordial. Antes de autorizar un despacho, "
        "debes verificar que el producto esté registrado en el catálogo ISP y que las condiciones del camión/vehículo "
        "cumplan la norma térmica (consultar_normas_bodega y evaluar_viabilidad_transporte).\n"
        "2. COMPORTAMIENTO ADAPTATIVO ANTE RIESGO: Si la temperatura del camión está fuera de rango "
        "(ej. >8°C para vacunas o >25°C para tópicos), NUNCA autorices el despacho. Debes cambiar de plan de inmediato: "
        "rechazar la carga y ejecutar obligatoriamente 'emitir_acta_cuarentena' para retener el lote y alertar al Director Técnico.\n"
        "3. PRODUCTO NO REGISTRADO (GUARDRAIL): Si al consultar el catálogo ISP el producto no existe o está fuera de norma, "
        "detén la operación y responde estrictamente que no figura en registros oficiales. No invoques herramientas de despacho.\n"
        "4. MEMORIA CONTINUA: Recuerda los productos y detalles mencionados en turnos previos de la conversación."
    )
)


def nodo_razonamiento_manager(estado: EstadoAgenteBodega) -> dict:
    """Nodo central ReAct: analiza la memoria episódica, planifica y decide las llamadas a herramientas."""
    llm_con_herramientas = instanciar_llm_agente()
    mensajes = [PROMPT_SISTEMA_MEDISCHIL] + estado["messages"]
    respuesta = llm_con_herramientas.invoke(mensajes)
    logger_agente.info(f"Nodo Razonador ejecutado para sesión: {estado['id_sesion']}")
    return {"messages": [respuesta]}


def nodo_ejecucion_herramientas(estado: EstadoAgenteBodega) -> dict:
    """Nodo determinista que intercepta y ejecuta las llamadas a herramientas."""
    ultimo_mensaje = estado["messages"][-1]
    nuevos_mensajes = []
    eventos = list(estado.get("trazabilidad_eventos", []))

    if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
        for tc in ultimo_mensaje.tool_calls:
            nombre = tc["name"]
            args = dict(tc["args"])
            call_id = tc["id"]

            # Inyección de rol de usuario autenticado si la herramienta requiere token_rol
            if "token_rol" in args or nombre in ("emitir_orden_despacho", "emitir_acta_cuarentena"):
                args["token_rol"] = estado.get("rol_usuario", "OPERARIO_BODEGA")

            logger_agente.info(f"Ejecutando herramienta: {nombre} | Args: {args}")

            if nombre in mapa_herramientas:
                try:
                    resultado = mapa_herramientas[nombre].invoke(args)
                except Exception as err:
                    resultado = json.dumps({"status": "ERROR_EJECUCION", "detalle": str(err)})
            else:
                resultado = json.dumps({"status": "ERROR", "mensaje": f"Herramienta '{nombre}' desconocida."})

            eventos.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": nombre,
                "resultado": resultado
            })

            nuevos_mensajes.append(ToolMessage(content=str(resultado), tool_call_id=call_id))

    return {"messages": nuevos_mensajes, "trazabilidad_eventos": eventos}


def enrutar_flujo_agente(estado: EstadoAgenteBodega) -> str:
    """Enrutador condicional: determina si continuar el ciclo ReAct o finalizar la tarea."""
    ultimo_mensaje = estado["messages"][-1]
    if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
        return "ejecutar_herramientas"
    return END


# Construcción del Grafo de Estados (LangGraph)
constructor_agente = StateGraph(EstadoAgenteBodega)
constructor_agente.add_node("razonador_manager", nodo_razonamiento_manager)
constructor_agente.add_node("ejecutar_herramientas", nodo_ejecucion_herramientas)

constructor_agente.add_edge(START, "razonador_manager")
constructor_agente.add_conditional_edges(
    "razonador_manager",
    enrutar_flujo_agente,
    {
        "ejecutar_herramientas": "ejecutar_herramientas",
        END: END
    }
)
constructor_agente.add_edge("ejecutar_herramientas", "razonador_manager")

# Grafo compilado del Agente de Bodega MedisChil S.A.
app_agente_medischil = constructor_agente.compile()


# =====================================================================
# 8. INTERFAZ OPERATIVA Y SESIONES MULTI-TURNO
# =====================================================================
class SesionOperativaBodega:
    """Gestiona una sesión conversacional con memoria de corto plazo y trazabilidad."""

    def __init__(self, id_sesion: str = "SESION-DEFAULT", rol_usuario: str = "OPERARIO_BODEGA"):
        self.id_sesion = id_sesion
        self.rol_usuario = rol_usuario
        self.estado: EstadoAgenteBodega = {
            "messages": [],
            "id_sesion": id_sesion,
            "rol_usuario": rol_usuario,
            "trazabilidad_eventos": []
        }

    def enviar_mensaje(self, texto_usuario: str) -> str:
        """Envía una instrucción al agente y acumula el historial episódico."""
        self.estado["messages"].append(HumanMessage(content=texto_usuario))
        resultado = app_agente_medischil.invoke(self.estado)
        self.estado = resultado
        respuesta_final = self.estado["messages"][-1].content
        return respuesta_final
