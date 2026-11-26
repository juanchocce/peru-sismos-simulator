# Perú Sismos Simulator (`peru-sismos-simulator`)

Plataforma integral de ingeniería de datos geoespaciales, sismotectónica cuantitativa, Machine Learning geofísico y simulación interactiva de riesgo sísmico para el Perú, indexada por **UBIGEO (6 dígitos)** y celdas hexagonales **Uber H3 (`res=5`)**.

---

## Inicio Rápido (App Interactiva Streamlit)

El repositorio incluye los datasets procesados (`data/processed/`, `data/interim/`), modelos entrenados (`models/`) y límites tectónicos PB2002 necesarios para ejecutar el simulador directamente tras clonar:

```bash
git clone https://github.com/juanchocce/peru-sismos-simulator.git
cd peru-sismos-simulator
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Capacidades Principales

1. **Simulador de Escenarios Sísmicos en Tiempo Real (ShakeMap Híbrido No Lineal + Vista 3D):**
   - Modelo GMPE LightGBM (`models/gmpe_lgbm.joblib`) con validación `GroupKFold` por evento sísmico, extrapolación física de magnitud para megaterremotos ($M_w \ge 8.0$) mediante pendiente Huber, y amortiguamiento geotécnico no lineal de amplificación de sitio ($\delta S_{2S}$).
   - Visualización 3D interactiva del hipocentro subterráneo y propagación de ondas P ($6.5\text{ km/s}$) y S ($3.8\text{ km/s}$) sobre los **1,893 distritos** del Perú (`territorial_multihazard_ubigeo.parquet`).
2. **Consultor Distrital y Auditoría FONDES / SIGRID:**
   - Probabilidad estadística multi-horizonte, polígonos oficiales INEI WFS (`inei_distritos.geojson`), evaluación bajo 5 escenarios (incluyendo simulación personalizada), guía de vulnerabilidad estructural y auditoría de brechas frente al padrón FONDES / SIGRID.
3. **Brechas y Lagunas Sísmicas Costeras (Monte Carlo & BPT):**
   - Balance de momento sísmico en 30 segmentos costeros de $0.5^\circ$ + 5 macro-bloques, simulación Monte Carlo ($N=10,000$) y probabilidad de recurrencia BPT a 30/50 años (`seismic_gaps_probabilistic.parquet`).
4. **Pronóstico Espacio-Temporal H3:**
   - Panel mensual estrictamente causal `2000–2026` sobre **8,205 hexágonos H3 (`res=5`)** (`h3_spatiotemporal_panel.parquet`) con modelo LightGBM validado mediante *walk-forward* y calibración Platt (`models/h3_forecaster_lgbm.joblib`).
5. **Escuela Sísmica, Historia 1552–2026 y Extensibilidad Multi-Riesgo:**
   - Guía didáctica interactiva para ciudadanos, corte 3D Wadati–Benioff sobre el contorno del Perú, catálogos IGP Histórico (`1552–1959`) e Instrumental (`1960–2026`), acelerometría ACELDAT (`2019–2025`), enriquecimiento USGS y zonas críticas GEOCATMIN con columnas reservadas `ext_*` para acoplar futuras amenazas.

---

## Estructura del Repositorio

```text
peru-sismos-simulator/
├── .streamlit/config.toml     # Configuración de tema Streamlit
├── app/
│   ├── streamlit_app.py       # Aplicación interactiva de 9 secciones en Streamlit
│   └── ui_theme.py            # Sistema visual Dual Claro/Oscuro, diccionarios y tablas centradas
├── data/
│   ├── raw/tectonic_plates/   # GeoJSON PB2002 requeridos por el simulador
│   ├── interim/               # Métricas JSON, polígonos INEI WFS, modelo slab 3D (.parquet/.geojson)
│   └── processed/             # Datasets maestros listos para la app y ML (.parquet)
├── docs/                      # Documentación técnica pública y tarjetas de modelos
├── models/                    # Modelos serializados (.joblib)
├── notebooks/                 # Exploración reproducible
├── reports/                   # Mapas HTML ejecutivos y figuras interactivas
├── src/
│   ├── ingestion/             # ETL IGP, ACELDAT, USGS, INEI WFS y FONDES/GEOCATMIN
│   ├── features/              # Distancias 3D, Wadati–Benioff, Gutenberg–Richter, lagunas y H3
│   ├── models/                # GMPE, corrección no lineal, BPT/MC, forecaster H3 e integración
│   └── viz/                   # Generación de mapas y tablero HTML ejecutivo
├── requirements.txt           # Dependencias Python
└── README.md
```

> **Nota sobre fuentes originales (`base_informativa/`):** Los archivos crudos de terceros (IGP, GEOCATMIN FileGDB, PDFs FONDES/CISMID) son de solo lectura local y están excluidos del control de versiones mediante `.gitignore`.

---

## Documentación Técnica y Reproducibilidad

- Documentación técnica integral y rigor estadístico: [`docs/DOCUMENTACION_TECNICA_COMPLETA.md`](docs/DOCUMENTACION_TECNICA_COMPLETA.md)
- Arquitectura end-to-end y orden completo de ejecución: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Tarjetas de modelos y validación anti-leakage: [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md)
- Auditoría técnica de fuentes y calidad de datos: [`docs/DATA_AUDIT_REPORT.md`](docs/DATA_AUDIT_REPORT.md)
