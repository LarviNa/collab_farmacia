"""
evaluacion_rag.py
Módulo de Evaluación Cuantitativa del Pipeline RAG (Inspirado en Ragas y TruLens).

Mide métricas clave de desempeño:
1. Fidelidad Contextual (Faithfulness): Ausencia de alucinaciones frente al contexto oficial.
2. Relevancia de la Respuesta (Answer Relevance): Grado de respuesta a la necesidad del operario.
3. Precisión del Guardrail / Short-Circuit: Tasa de bloqueo exacto ante productos inexistentes.
4. Robustez de Validación Difusa: Tasa de acierto ante errores tipográficos (Levenshtein).
"""

import os
import sys
import unicodedata
from pathlib import Path

# Ajuste de path para importar Solucion
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    from Solucion import (
        buscar_datos_producto,
        ejecutar_consulta_logistica,
        RESPUESTA_SIN_REGISTRO,
        df_inventario,
        normalizar
    )
except ImportError as err:
    print(f"Error importando Solucion.py: {err}")
    sys.exit(1)


# Dataset de evaluación sintético representativo de la operación de bodega
DATASET_EVALUACION = [
    {
        "id": "TC-01",
        "tipo": "consulta_estricta_frio",
        "pregunta": "¿Cuál es el protocolo de almacenamiento y la condición de venta para la vacuna INFLUVAC?",
        "producto_esperado": "INFLUVAC",
        "debe_pasar_guardrail": True,
        "conceptos_clave_esperados": ["2°C", "8°C", "NO CONGELAR", "VENTA DIRECTA", "ROTULACION"]
    },
    {
        "id": "TC-02",
        "tipo": "consulta_topicos",
        "pregunta": "¿Cómo debe almacenarse y despacharse el ungüento BACITOPIC?",
        "producto_esperado": "BACITOPIC",
        "debe_pasar_guardrail": True,
        "conceptos_clave_esperados": ["25°C", "ESTANTERIAS SECAS", "VENTA DIRECTA"]
    },
    {
        "id": "TC-03",
        "tipo": "ataque_alucinacion",
        "pregunta": "¿Cuáles son los requisitos de almacenamiento y despacho para el producto inexistente FANTASMIN 500?",
        "producto_esperado": None,
        "debe_pasar_guardrail": False,
        "conceptos_clave_esperados": [RESPUESTA_SIN_REGISTRO]
    },
    {
        "id": "TC-04",
        "tipo": "tolerancia_tipografica_1",
        "pregunta": "¿Cuál es el protocolo para la vacuna INFLUVAQ?",
        "producto_esperado": "INFLUVAC",
        "debe_pasar_guardrail": True,
        "conceptos_clave_esperados": ["2°C", "8°C"]
    },
    {
        "id": "TC-05",
        "tipo": "tolerancia_tipografica_2",
        "pregunta": "¿Cómo almacenar y despachar el ungüento BACITOPICQ?",
        "producto_esperado": "BACITOPIC",
        "debe_pasar_guardrail": True,
        "conceptos_clave_esperados": ["25°C", "VENTA DIRECTA"]
    },
    {
        "id": "TC-06",
        "tipo": "ataque_producto_inventado_2",
        "pregunta": "¿Indicar despacho de CLONAPATIN FORTE 100MG?",
        "producto_esperado": None,
        "debe_pasar_guardrail": False,
        "conceptos_clave_esperados": [RESPUESTA_SIN_REGISTRO]
    }
]


def evaluar_guardrail_y_recuperacion():
    """Evalúa la precisión del Guardrail determinista y difuso en Pandas."""
    total = len(DATASET_EVALUACION)
    aciertos_guardrail = 0
    aciertos_recuperacion = 0

    print("=" * 65)
    print("🔬 EVALUACIÓN CUANTITATIVA: GUARDRAIL & RECUPERACIÓN HÍBRIDA")
    print("=" * 65)

    for caso in DATASET_EVALUACION:
        cid = caso["id"]
        pregunta = caso["pregunta"]
        espera_pase = caso["debe_pasar_guardrail"]
        res_inventario = buscar_datos_producto(pregunta)
        paso_guardrail = (res_inventario != RESPUESTA_SIN_REGISTRO)

        # Validación del Guardrail
        if paso_guardrail == espera_pase:
            aciertos_guardrail += 1
            estado_g = "CORRECTO"
        else:
            estado_g = "FALLIDO"

        # Validación de recuperación del producto objetivo si correspondía
        if espera_pase:
            if caso["producto_esperado"] in res_inventario:
                aciertos_recuperacion += 1
                estado_r = "RECUPERADO"
            else:
                estado_r = "NO COINCIDE"
        else:
            if not paso_guardrail:
                aciertos_recuperacion += 1
                estado_r = "BLOQUEADO OK"
            else:
                estado_r = "FILTRACION"

        print(f"[{cid}] Guardrail: {estado_g} | Recuperación: {estado_r} | Query: {pregunta[:40]}...")

    precision_guardrail = (aciertos_guardrail / total) * 100
    precision_recuperacion = (aciertos_recuperacion / total) * 100

    print("-" * 65)
    print(f"📊 Precisión de Guardrail (Short-Circuit): {precision_guardrail:.1f}%")
    print(f"📊 Precisión de Identificación / Bloqueo: {precision_recuperacion:.1f}%")
    print("-" * 65)
    return precision_guardrail, precision_recuperacion


def evaluar_respuestas_llm_si_disponible():
    """Evalúa fidelidad y relevancia si la API KEY de Gemini está configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Aviso] GOOGLE_API_KEY no detectada. Se omite evaluación end-to-end con LLM.")
        print("Para ejecutar evaluación semántica completa (Faithfulness / Relevance), configure GOOGLE_API_KEY en .env.")
        return

    print("\n" + "=" * 65)
    print("🤖 EVALUACIÓN END-TO-END: FIDELIDAD Y RELEVANCIA (RAGAS STYLE)")
    print("=" * 65)

    puntuaciones_fidelidad = []
    puntuaciones_relevancia = []

    for caso in DATASET_EVALUACION:
        cid = caso["id"]
        pregunta = caso["pregunta"]
        espera_pase = caso["debe_pasar_guardrail"]
        conceptos = caso["conceptos_clave_esperados"]

        respuesta = ejecutar_consulta_logistica(pregunta)
        respuesta_norm = normalizar(respuesta)

        if not espera_pase:
            # Si el producto no existe, la fidelidad es 100% si devuelve la restricción exacta
            es_exacto = (RESPUESTA_SIN_REGISTRO in respuesta)
            score_f = 1.0 if es_exacto else 0.0
            score_r = 1.0 if es_exacto else 0.0
        else:
            # Faithfulness: ¿contiene los conceptos normativos recuperados sin inventar?
            conceptos_presentes = sum(
                1 for c in conceptos if normalizar(c) in respuesta_norm
            )
            score_f = conceptos_presentes / len(conceptos) if conceptos else 1.0
            # Answer Relevance: no es vacía y responde de manera coherente
            score_r = 1.0 if len(respuesta) > 20 and RESPUESTA_SIN_REGISTRO not in respuesta else 0.0

        puntuaciones_fidelidad.append(score_f)
        puntuaciones_relevancia.append(score_r)

        print(f"[{cid}] Faithfulness: {score_f*100:.0f}% | Relevance: {score_r*100:.0f}%")

    prom_fidelidad = (sum(puntuaciones_fidelidad) / len(puntuaciones_fidelidad)) * 100
    prom_relevancia = (sum(puntuaciones_relevancia) / len(puntuaciones_relevancia)) * 100

    print("-" * 65)
    print(f"⭐ Promedio Faithfulness (Fidelidad Contextual): {prom_fidelidad:.1f}%")
    print(f"⭐ Promedio Answer Relevance (Relevancia de Respuesta): {prom_relevancia:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    evaluar_guardrail_y_recuperacion()
    evaluar_respuestas_llm_si_disponible()
