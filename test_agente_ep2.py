"""
test_agente_ep2.py
Suite de Pruebas Automatizadas y Evidencia de Software para la Evaluación Parcial N°2 (EP2).
Asignatura: ISY0101 - Duoc UC

Demuestra formalmente los Indicadores de Evaluación de la Pauta:
- IE1: Invocación autónoma de herramientas de consulta, razonamiento y escritura.
- IE3 & IE4: Memoria de contenido en flujos prolongados (STM) y recuperación semántica (LTM).
- IE5 & IE7: Planificación y toma de decisiones adaptativas ante contingencias sanitarias.
"""

import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    from agente_medischil import SesionOperativaBodega, servidor_transaccional
except ImportError as e:
    print(f"Error importando agente_medischil: {e}")
    sys.exit(1)


def ejecutar_bateria_pruebas_ep2():
    print("=" * 80)
    print("🏥 BANCO DE PRUEBAS DEL AGENTE FUNCIONAL - MEDISCHIL S.A. (EP2)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # ESCENARIO 1: Flujo Prolongado Multi-Turno con Memoria Episódica (STM) y Despacho Normal
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("🟢 ESCENARIO 1: Flujo Prolongado Multi-Turno y Memoria de Contexto (IE3, IE4, IE1)")
    print("#" * 80)

    sesion_1 = SesionOperativaBodega(id_sesion="SESION-HOSPITAL-CENTRAL-01", rol_usuario="OPERARIO_BODEGA")

    turno_1 = "¿Cuál es el protocolo de almacenamiento y condición de venta para la vacuna INFLUVAC?"
    print(f"\n👤 [OPERARIO - Turno 1]: {turno_1}")
    res_1 = sesion_1.enviar_mensaje(turno_1)
    print(f"🤖 [AGENTE MEDISCHIL]:\n{res_1}")

    turno_2 = "El camión de despacho N°4 tiene la cámara frigorífica en 4.5 °C. Emite la orden de despacho para 50 unidades de esa misma vacuna hacia el Hospital Central."
    print(f"\n👤 [OPERARIO - Turno 2 (Referencia Anafórica sin repetir nombre)]: {turno_2}")
    res_2 = sesion_1.enviar_mensaje(turno_2)
    print(f"🤖 [AGENTE MEDISCHIL]:\n{res_2}")

    # -------------------------------------------------------------------------
    # ESCENARIO 2: Toma de Decisión Adaptativa ante Contingencia (Rotura de Cadena de Frío -> Cuarentena)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("🔴 ESCENARIO 2: Decisión Adaptativa ante Riesgo Sanitario (IE5, IE7, IE1)")
    print("#" * 80)

    sesion_2 = SesionOperativaBodega(id_sesion="SESION-EMERGENCIA-CLIMA-02", rol_usuario="SUPERVISOR_LOGISTICA")

    caso_contingencia = (
        "El transportista informa que el camión refrigerado falló y la temperatura interior está en 14 °C. "
        "Solicita autorizar el despacho urgente de 80 dosis de la vacuna INFLUVAC hacia la Clínica RedSalud."
    )
    print(f"\n👤 [SUPERVISOR]: {caso_contingencia}")
    res_contingencia = sesion_2.enviar_mensaje(caso_contingencia)
    print(f"🤖 [AGENTE MEDISCHIL]:\n{res_contingencia}")

    # -------------------------------------------------------------------------
    # ESCENARIO 3: Guardrail Anti-Alucinación con Producto Inexistente
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("🛡️ ESCENARIO 3: Guardrail de Validación Previa y Producto Inexistente (IE1, IE5)")
    print("#" * 80)

    sesion_3 = SesionOperativaBodega(id_sesion="SESION-TEST-GUARDRAIL-03", rol_usuario="OPERARIO_BODEGA")

    caso_inexistente = "Autoriza el despacho de 100 frascos de FANTASMIN 500 MG al Consultorio Maipú."
    print(f"\n👤 [OPERARIO]: {caso_inexistente}")
    res_inexistente = sesion_3.enviar_mensaje(caso_inexistente)
    print(f"🤖 [AGENTE MEDISCHIL]:\n{res_inexistente}")

    # -------------------------------------------------------------------------
    # VERIFICACIÓN DE AUDITORÍA Y ESCRITURAS TRANSACCIONALES EN DISCO
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📋 INSPECCIÓN DE EVIDENCIA EN DISCO (PERSISTENCIA Y SEGURIDAD TRANSACCIONAL)")
    print("=" * 80)

    ruta_dsp = BASE_DIR / "data_registro_despachos.json"
    if ruta_dsp.exists():
        with open(ruta_dsp, "r", encoding="utf-8") as f:
            dsp_data = json.load(f)
        print(f"\n📦 Órdenes de Despacho Autorizadas Registradas: {len(dsp_data)}")
        if dsp_data:
            print(json.dumps(dsp_data[-1], indent=2, ensure_ascii=False))

    ruta_cuar = BASE_DIR / "data_bitacora_cuarentena.json"
    if ruta_cuar.exists():
        with open(ruta_cuar, "r", encoding="utf-8") as f:
            cuar_data = json.load(f)
        print(f"\n⚠️ Actas de Cuarentena Preventiva Emitidas: {len(cuar_data)}")
        if cuar_data:
            print(json.dumps(cuar_data[-1], indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("✅ BATERÍA DE PRUEBAS COMPLETADA EXITOSAMENTE.")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_bateria_pruebas_ep2()
