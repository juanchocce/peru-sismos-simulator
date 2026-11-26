# MODEL CARDS — Sistema Sismotectónico y de Riesgo Sísmico del Perú

Este documento resume el alcance operativo, arquitectura, supuestos geofísicos, validación anti-leakage y limitaciones explícitas de los 3 modelos entrenados del sistema.

---

## Model Card 1 — Modelo A: GMPE con Machine Learning y Efecto de Sitio ($\delta S_{2S}$)

- **Identificador / Artefacto:** `gmpe_lgbm.joblib` (`src/models/train_gmpe.py`)
- **Objetivo:** Predecir la aceleración máxima horizontal media geométrica del suelo ($\log_{10}(\text{PGA}_{H,\text{geom}})$ en Gal = $\text{cm/s}^2$) y vertical ($\log_{10}(\text{PGA}_V)$) para cualquier escenario sísmico en Perú, en dos modos:
  1. `model_no_site`: generalizable a cualquier coordenada del territorio nacional (sin requerir estación previa).
  2. `model_with_site`: enriquecido con el término empírico regularizado de sitio $\delta S_{2S}$ para las 277 estaciones de la red ACELDAT/IGP.
- **Datos de entrenamiento:** `data/processed/aceldat_merged.parquet` (`17,794` registros estación-evento deduplicados sobre `956` eventos únicos, `2019–2025`), enriquecido con geometría 3D del slab Wadati–Benioff y conteos distritales GEOCATMIN.
- **Validación sin fuga:**
  - `GroupKFold(n_splits=5)` agrupado estrictamente por `event_id` (cero eventos compartidos entre train y test).
  - Evaluación *out-of-time*: Train `2019–2024` (`14,718` filas) → Test `2025` (`3,076` filas).
  - Término de sitio $\delta S_{2S}$ estimado mediante nested 4-fold `GroupKFold` interno + Empirical Bayes shrinkage ($\lambda = 5.0$) estrictamente dentro de cada partición de entrenamiento.
- **Métricas Clave (`log10_pga_h_geom`):**
  - `GroupKFold-5`: `RMSE = 0.2829` ($R^2 = 0.8070$, $\tau = 0.235$, $\phi = 0.193$, factor geométrico `1.65×`) vs Baseline empírico Huber (`RMSE = 0.3810`, $R^2 = 0.6498$).
  - `Holdout 2025`: `RMSE = 0.2943` ($R^2 = 0.7806$) vs Baseline (`RMSE = 0.3894`, $R^2 = 0.6160$).
- **Supuestos Geofísicos y Limitaciones Explícitas:**
  1. **Extrapolación en megaterremotos ($M_w \ge 8.0$):** ACELDAT 2019–2025 no contiene megaterremotos de subducción $M_w > 8.0$ en campo cercano ($R < 30\text{ km}$). Los árboles de decisión (GBDT) tienden a saturar en los extremos fuera del rango observado; para escenarios deterministas $M_w \ge 8.0$ (ej. ruptura de Lima 1746), el GMPE ML debe combinarse o auditarse contra leyes paramétricas de subducción (BC Hydro / Montalva) y efectos de fuente finita ($R_{\text{rup}}$).
  2. **Ausencia de $V_{S30}$ instrumental nacional:** Al no existir mapa nacional de $V_{S30}$, `model_no_site` usa proxies geomorfológicos (GEOCATMIN/distancia a fosa); la amplificación local fina solo está caracterizada en las 277 estaciones con registros (`aceldat_station_site_factors.parquet`).
  3. **No linealidad de suelos:** Los factores $\delta S_{2S}$ reflejan respuesta lineal/débil a moderada; bajo aceleraciones extremas ($> 200\text{ Gal}$), suelos blandos pueden experimentar degradación no lineal y amortiguamiento adicional.

---

## Model Card 2 — Modelo B: Lagunas Sísmicas Probabilísticas (Monte Carlo + BPT + HDBSCAN)

- **Identificador / Artefacto:** `data/processed/seismic_gaps_probabilistic.parquet` (`src/models/model_seismic_gaps.py`)
- **Objetivo:** Cuantificar la distribución probabilística (P10/P50/P90) del déficit de momento sísmico ($\Delta M_0$) y la Magnitud Equivalente Potencial ($M_{w,\text{pot}}$), junto con probabilidades condicionales de ruptura $M_w \ge 8.0$ a 30 y 50 años (modelo de renovación *Brownian Passage Time*, BPT) y densidad de liberación moderna (HDBSCAN).
- **Datos de entrada:** Catálogo histórico IGP (`1552–1959`) + instrumental (`1960–2026`), geometría continua de fosa PB2002 (`NZ\SA` + `NZ\AP` + `NZ\ND`) y tasa de convergencia real por segmento.
- **Métricas / Resultados Clave:**
  - **Laguna Central Lima–Chincha (`-11.5°` a `-13.5°`, desde `1746.8`, `280.1 años`):** $M_{w,\text{pot}}$ P50 = **`8.62`** `[P10 8.54 – P90 8.69]`; BPT $P_{30\text{yr}} = 36.5\%$ ($\alpha=0.5$) / $63.5\%$ ($\alpha=0.3$); $P_{50\text{yr}} = 53.1\%$ ($\alpha=0.5$).
  - **Sub-bloque Sur Ilo–Tacna (`-17.0°` a `-18.5°`, desde `1869.6`, `157.2 años`):** $M_{w,\text{pot}}$ P50 = **`8.39`** `[P10 8.31 – P90 8.47]`; BPT $P_{30\text{yr}} = 34.0\%$ ($\alpha=0.5$) / $51.2\%$ ($\alpha=0.3$); $P_{50\text{yr}} = 50.5\%$ ($\alpha=0.5$).
- **Supuestos Geofísicos y Limitaciones Explícitas:**
  1. **Acoplamiento intersísmico ($\chi$) homogéneo en simulación:** El muestreo $\chi \sim U(0.50, 0.85)$ no incorpora inversión geodésica GNSS espacialmente variable (e.g. parches de deslizamiento asísmico *creeping* en el norte de Perú o frente a la Dorsal de Nazca ~$-15^\circ\text{S}$). En el Bloque Norte, un acoplamiento real menor ($\chi < 0.3$) reduciría el déficit efectivo.
  2. **Ruptura de segmentos independientes vs multi-segmento:** El cálculo BPT usa tiempo medio de recurrencia $T_r \in [150, 300]\text{ años}$ acotado físicamente; una ruptura en cascada puede liberar múltiples segmentos simultáneamente o de forma parcial (como 1940/1966/2007).
  3. **Incertidumbre histórica pre-instrumental:** Los epicentros y magnitudes previas a 1900 poseen incertidumbre de localización ($\pm 50\text{–}100\text{ km}$) y magnitud ($\sigma_M = 0.25$ propagada en Monte Carlo).
  4. **No es predicción determinista de fecha:** Las probabilidades BPT son tasas condicionales de peligro a escala multidecadal (30–50 años) para planificación territorial, normativa E.030 y priorización de inversión pública.

---

## Model Card 3 — Modelo C: Pronóstico Probabilístico Espacio-Temporal en Grilla H3 (Res 5)

- **Identificador / Artefacto:** `models/h3_forecaster_lgbm.joblib` y `data/processed/h3_forecast_latest.parquet` (`src/models/train_h3_forecaster.py`)
- **Objetivo:** Estimar probabilidades calibradas de ocurrencia de al menos un sismo $M \ge 4.5$ (en 90 y 365 días) y $M \ge 5.0$ (en 90 días) en cada una de las `8,205` celdas hexagonales H3 resolución 5 (~252 km²) que cubren Perú y su margen costero (150 km).
- **Datos y Validación sin Fuga:**
  - `data/processed/h3_spatiotemporal_panel.parquet` (`2,650,215` filas × `53` cols, cortes mensuales `2000-01-01` a `2026-11-01`).
  - **4 Splits Expanding-Window Walk-Forward (`2018–2025`)** con **purge gap obligatorio** (4 meses para 90d, 12 meses para 365d), tasa espacial expansiva estrictamente rezagada y calibración Platt ajustada únicamente sobre los 24 meses finales del train.
- **Métricas Clave (Promedio 4 Splits Walk-Forward):**
  - `target_m45_90d`: `XGBoost` **`ROC-AUC = 0.8527`**, **`PR-AUC = 0.0389`**, **`BSS = +0.0789`** (`LightGBM` `PR-AUC = 0.0390`, `BSS = +0.0772`) vs Poisson Estacionario (`PR-AUC = 0.0324`, `ROC-AUC = 0.8160`) y ETAS Paramétrico (`PR-AUC = 0.0330`, `BSS = +0.0636`).
  - `target_m45_365d`: `XGBoost` **`PR-AUC = 0.1168`**, **`ROC-AUC = 0.8504`**, **`BSS = +0.2026`** vs Poisson (`PR-AUC = 0.0989`) y ETAS (`BSS = +0.1537`).
  - `target_m50_90d`: `XGBoost` **`ROC-AUC = 0.8348`**, **`BSS = +0.0210`** vs Poisson (`ROC-AUC = 0.7110`).
- **Supuestos Geofísicos y Limitaciones Explícitas:**
  1. **Naturaleza estocástica de los terremotos:** El modelo estima **tasas probabilísticas espaciales y de agrupamiento espacio-temporal (réplicas/enjambres tipo ETAS + estructura tectónica 3D)**, NO predice determinísticamente el día ni la hora de sismos principales independientes (*mainshocks*).
  2. **Eventos raros (`M >= 5.0` en 90 días, 0.24% positivos):** Aunque mejora sustancialmente la discriminación espacial (`ROC-AUC 0.835`) y la calibración (`Log-Loss 0.0097`), el PR-AUC absoluto (`~0.012`) refleja el límite físico de predictibilidad de sismos moderados-fuertes a corto plazo en celdas de ~252 km².
  3. **Cambio temporal de completitud ($M_c$):** A partir de 2020 la red IGP bajó $M_c$ de `4.7` a `4.2`; los targets evaluados ($M \ge 4.5$ y $M \ge 5.0$) permanecen cerca o por encima de la completitud moderna para minimizar sesgo de detección, pero la tasa histórica de `2000–2019` presenta ligera deriva temporal compensada mediante calibración Platt sobre la sub-ventana reciente.
