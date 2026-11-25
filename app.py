from datetime import datetime
import hashlib
import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Sistema Predictivo de Mantenimiento Industrial",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def cargar_modelo():
    """Carga el modelo serializado una sola vez en memoria."""
    return joblib.load("modelo_mantenimiento.pkl")


@st.cache_data(ttl=300, show_spinner=False)
def generar_informe_groq(
    key_fingerprint: str,
    api_key: str,
    estado: int,
    confianza: float,
    temperatura: float,
    vibracion: float,
    horas: int,
    presion: float,
) -> tuple[str, str]:
    """Detecta dinámicamente un modelo activo en Groq y genera el dictamen u Orden de Trabajo."""
    _ = key_fingerprint  # Diferencia caché si cambia la clave sin exponerla en UI
    client = Groq(api_key=api_key)

    modelos_activos = sorted(
        m.id
        for m in client.models.list().data
        if not any(
            x in m.id.lower()
            for x in ["whisper", "guard", "tts", "playai", "embed", "orpheus", "allam"]
        )
    )
    if not modelos_activos:
        raise RuntimeError("No se encontraron modelos de chat activos en Groq API.")

    modelo_groq = next(
        (
            m
            for pref in ("llama", "gpt-oss-120b", "gpt-oss-20b", "qwen")
            for m in modelos_activos
            if pref in m.lower()
        ),
        modelos_activos[0],
    )

    system_prompt = (
        "Eres el Jefe de Mantenimiento Industrial de una planta de manufactura automatizada. "
        "Tu función es emitir dictámenes técnicos precisos y Órdenes de Trabajo estructuradas "
        "en Markdown a partir de telemetría de sensores y diagnósticos de Machine Learning. "
        "Responde siempre en español técnico profesional."
    )

    telemetria_txt = (
        f"- Temperatura: {temperatura} °C\n"
        f"- Vibración: {vibracion} mm/s\n"
        f"- Horas de operación: {horas} hrs\n"
        f"- Presión: {presion} PSI\n"
        f"- Estado clasificado por ML: {estado} (Confianza: {confianza:.1f}%)"
    )

    if estado == 0:
        user_prompt = (
            "La maquinaria se encuentra en Estado 0 (Operación Normal).\n"
            f"Telemetría actual:\n{telemetria_txt}\n\n"
            "Emite un dictamen breve de monitoreo preventivo de rutina (máximo un párrafo corto "
            "y 2 recomendaciones puntuales de inspección rutinaria) confirmando estabilidad operativa."
        )
    else:
        prefijo = "OT-PREV" if estado == 1 else "OT-CRIT"
        prioridad = (
            "Mantenimiento Preventivo Programado (Prioridad Media-Alta)"
            if estado == 1
            else "URGENTE — Fallo Crítico Inminente (Requiere Protocolo de Paro Inmediato)"
        )
        accion_paro = (
            "La Acción Técnica 1 DEBE iniciar con el protocolo de paro inmediato de línea y bloqueo/etiquetado (LOTO)."
            if estado == 2
            else "Incluye ajustes, lubricación, alineación o reemplazo preventivo de componentes con desgaste."
        )
        folio_sugerido = f"{prefijo}-{datetime.now().strftime('%Y%m%d')}-{int(horas):04d}"

        user_prompt = (
            f"Se requiere generar una Orden de Trabajo Técnico en Markdown.\n"
            f"Nivel de severidad: {prioridad}\n"
            f"Telemetría actual:\n{telemetria_txt}\n\n"
            "Estructura obligatoriamente el documento en Markdown de forma concisa y completa con estas 4 secciones:\n"
            f"1. **Folio**: Usa `{folio_sugerido}` indicando prioridad y fecha.\n"
            "2. **Diagnóstico**: Análisis técnico breve (máximo 5 líneas) de las lecturas fuera de rango.\n"
            f"3. **3 Acciones Técnicas**: Exactamente 3 pasos numerados y concretos. {accion_paro}\n"
            "4. **Equipo de Seguridad (EPP)**: Lista específica del EPP obligatorio para intervenir el equipo."
        )

    respuesta = client.chat.completions.create(
        model=modelo_groq,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    contenido = respuesta.choices[0].message.content or ""
    return modelo_groq, contenido


st.title("⚙️ Sistema Híbrido de Mantenimiento Predictivo Industrial")
st.markdown(
    "Monitoreo de telemetría en tiempo real, clasificación multiclase con Machine Learning "
    "y generación automática de Órdenes de Trabajo con IA Generativa (Groq)."
)

# Controles interactivos en la barra lateral
st.sidebar.header("🎛️ Telemetría de Sensores")
temperatura = st.sidebar.slider(
    "Temperatura (°C)", min_value=40.0, max_value=120.0, value=60.0, step=0.5
)
vibracion = st.sidebar.slider(
    "Vibración (mm/s)", min_value=0.0, max_value=12.0, value=1.5, step=0.1
)
horas = st.sidebar.slider(
    "Horas de Operación", min_value=0, max_value=5000, value=300, step=10
)
presion = st.sidebar.slider(
    "Presión (PSI)", min_value=70.0, max_value=170.0, value=92.0, step=0.5
)

# Tarjetas visuales de lecturas actuales
st.subheader("📊 Lecturas Actuales de Maquinaria")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col1.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

st.divider()

# Inferencia con el modelo de Machine Learning
modelo = cargar_modelo()
entrada_df = pd.DataFrame(
    [[temperatura, vibracion, horas, presion]],
    columns=["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"],
)

estado_predicho = int(modelo.predict(entrada_df)[0])
probabilidades = modelo.predict_proba(entrada_df)[0]
confianza = probabilidades[estado_predicho] * 100

st.subheader("🔍 Diagnóstico del Modelo Predictivo")

if estado_predicho == 0:
    st.success(
        f"🟢 Estado 0: Operación Normal — Parámetros dentro de rango estable (Probabilidad: {confianza:.1f}%)"
    )
elif estado_predicho == 1:
    st.warning(
        f"🟡 Estado 1: Mantenimiento Preventivo — Desgaste moderado; requiere revisión programada (Probabilidad: {confianza:.1f}%)"
    )
else:
    st.error(
        f"🔴 Estado 2: Fallo Crítico — Riesgo inminente de avería; requiere paro inmediato (Probabilidad: {confianza:.1f}%)"
    )

with st.expander("Ver distribución completa de probabilidades (predict_proba)"):
    etiquetas = [
        "0 - Operación Normal",
        "1 - Mantenimiento Preventivo",
        "2 - Fallo Crítico",
    ]
    for idx, etiqueta in enumerate(etiquetas):
        pct = float(probabilidades[idx])
        st.write(f"**{etiqueta}**: {pct * 100:.1f}%")
        st.progress(pct)

st.divider()

# Módulo C y D: Generación de Orden de Trabajo con Groq API y manejo resiliente de secretos
st.subheader("🤖 Dictamen Técnico / Orden de Trabajo (Groq IA)")

if st.button("🔄 Regenerar informe con IA"):
    generar_informe_groq.clear()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key or not groq_api_key.strip() or "PEGA_" in groq_api_key:
        raise ValueError(
            "Configura tu clave real en `.streamlit/secrets.toml` (GROQ_API_KEY) o en los Secrets de Streamlit Cloud."
        )

    fingerprint = hashlib.sha256(groq_api_key.encode("utf-8")).hexdigest()[:8]
    with st.spinner("Consultando Jefe de Mantenimiento IA en Groq..."):
        modelo_usado, informe_md = generar_informe_groq(
            fingerprint,
            groq_api_key.strip(),
            estado_predicho,
            confianza,
            temperatura,
            vibracion,
            horas,
            presion,
        )

    st.caption(f"Modelo Groq detectado dinámicamente: `{modelo_usado}`")
    st.markdown(informe_md)

except Exception as err:
    st.info(
        "⚠️ **Módulo de IA Generativa en espera o no disponible temporalmente.** "
        "El diagnóstico del modelo de Machine Learning sigue completamente operativo.\n\n"
        f"**Detalle técnico:** `{err}`"
    )
