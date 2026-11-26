# Documentación Técnica Completa y Rigor Estadístico — Observatorio Sísmico y Multi-Amenaza del Perú

> **Plataforma:** Observatorio Sísmico y Multi-Amenaza del Perú (`peru-sismos-simulator`)  
> **Alcance territorial:** 1,893 distritos oficiales del Perú (100% cobertura UBIGEO de 6 dígitos) y 8,205 celdas hexagonales Uber H3 (`res=5`, WGS84 `EPSG:4326`).  
> **Principio rector:** Cero fugas de información (*zero data leakage*), trazabilidad íntegra desde fuentes inmutables (`base_informativa/` en solo lectura) hasta predicciones calibradas, y comunicación científica honesta en español claro.

---

## 1. Inventario Oficial de Fuentes y Fechas Exactas de Corte (UTC)

Todas las series temporales internas se almacenan en formato **UTC ISO-8601 con precisión `datetime64[us, UTC]`**, indispensable para abarcar eventos históricos desde el año `1552` sin desbordar el límite de `datetime64[ns]` (`1677–2262`).

| Fuente / Archivo Original (`base_informativa/` o API) | Entidad Emisora | Fecha/Hora Inicio (UTC) | Fecha/Hora Fin (UTC) | Registros Crudos | Registros Limpios / Integrados | Rol en la Plataforma |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| `datos-sismicos_Historico_1471-01-01_1959-12-31.xlsx` | Instituto Geofísico del Perú (IGP) | `1552-07-02 10:30:00` | `1959-12-02 07:02:52` | 1,098 | 1,098 | Catálogo histórico homogeneizado a $M_w$ (Scordilis 2006) para balance secular de momento |
| `datos-sismicos_Instrumental_1960-01-01_2026-11-22.xlsx` | Instituto Geofísico del Perú (IGP) | `1960-01-13 15:40:34` | `2026-11-22 05:51:57` | 25,878 | 25,878 | Catálogo maestro moderno (completitud $M_c$, Wadati–Benioff 3D, panel causal H3) |
| `IGP_ACELDAT_Peru_2019-2025_Dataset_0.csv` | IGP — Red Acelerográfica Nacional (ACELDAT) | `2019-01-03 21:11:48` | `2025-12-28 23:19:55` | 18,313 | 17,794 (956 sismos, 277 est.) | Entrenamiento GMPE (Modelo A) y factores empíricos de amplificación de suelo ($\delta S_{2S}$) |
| API FDSN USGS (`M ≥ 4.5`, caja Perú) | U.S. Geological Survey (USGS) | `1960-01-15 09:30:21` | `2026-11-22 21:35:31` | 7,524 | 4,237 emparejados a IGP | Enriquecimiento con *gap* azimutal, RMS, MMI ShakeMap (391), alertas PAGER (133) y tensores de momento (554) |
| `FONDES_relacion_distritos_nov26.pdf` | CENEPRED / SIGRID — Padrón FONDES | Vigencia corte `05/11/2026` | Vigencia corte `05/11/2026` | 1,893 distritos | 1,893 distritos (100% UBIGEO) | Marco oficial distrital y auditoría de subestimación sísmica frente a clasificación multi-peligro |
| `geocatmin_datos.zip` (`Peligros_Geologicos` y `Zonas_Criticas`) | INGEMMET — GEOCATMIN | Inventario histórico vigente | Corte descarga 2026 | 37,023 peligros / 2,177 zonas críticas | 35,673 peligros / 2,177 zonas críticas asignados a UBIGEO | Amenazas secundarias (huaicos, deslizamientos, caída de rocas) e índice multi-amenaza |
| `tectonicplates-master.zip` (PB2002) | Bird (2003) — Modelo Global de Placas | Modelo tectónico de referencia | Vigente | 3 segmentos de fosa (`NZ\SA`, `NZ\AP`, `NZ\ND`) | 6,455 vértices densificados (2 km) | Geometría 3D de fosa peruano-chilena, velocidad de convergencia y edad de placa oceánica |
| Servicios WFS INEI (`ig_departamento`, `ig_provincia`, `ig_distrito`) | INEI — GeoServidor Interoperabilidad | Cartografía oficial vigente | Descarga WFS (`2026-11-24 UTC`) | 25 dptos. / 196 prov. / 1,890 dist. | Polígonos simplificados (`0.002°`, `EPSG:4326`, 6.49 MB total) | Visualización cartográfica oficial en mapa nacional y consultor distrital |

> **Nota de claridad ciudadana sobre las fechas de corte:** Todos los cálculos de probabilidad y mapas reflejan el estado exacto de la sismicidad peruana hasta el **22 de noviembre de 2026** (último corte del catálogo instrumental IGP) y las mediciones de aceleración de suelos hasta el **28 de diciembre de 2025** (ACELDAT).

---

## 2. Auditoría de Calidad, Anomalías Detectadas y Armonización (ETL)

Durante la auditoría inicial y el pipeline ETL se inspeccionó programáticamente cada archivo crudo sin asumir nombres de columnas ni formatos. A continuación se documentan las **6 anomalías críticas** descubiertas y su remediación exacta:

1. **Codificación y delimitador en ACELDAT (`cp1252` + `;` + 10 columnas fantasma):**
   - El CSV oficial de ACELDAT viene en codificación `Windows-1252` separado por punto y coma, contiene **10 columnas vacías `Unnamed:*`** y pierde los ceros iniciales en `UBIGEO` y `HORA_EVENTO`.
   - **Solución:** Lectura explícita `encoding="cp1252", sep=";"`, descarte automático de `Unnamed:*`, normalización estricta a 6 dígitos (`zfill(6)`) para `ubigeo` y `hora_evento_str`, y parser dual para `FECHA_EVENTO` (`%Y%m%d` y `%m/%d/%Y`).
2. **Desfase horario `UTC-5` en ACELDAT (rescate de `0.50%` a `92.80%`):**
   - Aunque el catálogo IGP está en UTC, las marcas de tiempo de ACELDAT estaban registradas en **hora local del Perú (`UTC-5`)**. El cruce directo sin ajuste solo emparejaba **92 de 18,313 filas (`0.50%`)**.
   - **Solución:** Al aplicar el desplazamiento `+5 horas` hacia UTC (tolerancia $\pm 120\text{ s}$), el emparejamiento saltó a **16,994 filas (`92.80%`)** sobre **913 eventos únicos**, con una mediana de error temporal $|\Delta t| = 0.0\text{ s}$ (percentil 95: `2.0 s`).
3. **Inversión de día/mes (`YYYYDDMM`) en el residuo ACELDAT 2025 (rescate hasta `98.33%`):**
   - De las `1,319` filas inicialmente no emparejadas, **1,014 correspondían a 2025** y todas tenían el campo de día $\le 12$.
   - **Solución:** La auditoría (`src/features/audit_aceldat_unmatched.py`) probó el intercambio día/mes (`DD ↔ MM`) manteniendo `UTC-5 → UTC (+5h)`: los **1,014 registros (100% del residuo 2025, 43 sismos adicionales)** emparejaron con **error temporal mediano de `0.0 s` y percentil 95 de `0.0 s`**, elevando el dataset consolidado a **17,794 registros acelerográficos sobre 956 eventos únicos y 277 estaciones** (tasa consolidada de `98.33%` de las filas con evento catalogado).
4. **Continuidad de la fosa de subducción en PB2002 (`NZ\SA` + `NZ\AP` + `NZ\ND`):**
   - Filtrar únicamente el código `NZ\SA` dejaba un hueco artificial frente al norte y extremo sur del Perú porque el modelo PB2002 subdivide el margen andino en tres placas adyacentes: Sudamérica (`NZ\SA`: 17,148 sismos), Altiplano (`NZ\AP`: 8,195 sismos) y Andes del Norte (`NZ\ND`: 535 sismos).
   - **Solución:** Unión geodésica de los tres segmentos y densificación cada `2 km` sobre el elipsoide WGS84 (**6,455 vértices** indexados en `cKDTree` 3D).
5. **Homogeneización histórica de magnitudes y valores faltantes (`'-'`):**
   - El catálogo histórico IGP (1552–1959) codifica valores faltantes como guion literal `'-'` (59 profundidades ausentes) y mezcla escalas $m_b$ (892 eventos), $M_s$ (194) y $M_w$ (12).
   - **Solución:** Conversión explícita `'-' → NaN` y homogeneización empírica a Momento $M_w$ mediante las relaciones de **Scordilis (2006)** antes de integrar el momento sísmico acumulado.
6. **Reparación estructural del padrón PDF FONDES / SIGRID:**
   - Al extraer la tabla oficial de 1,893 distritos mediante `pdfplumber`, se detectaron y repararon **5 celdas desbordadas** por saltos de línea en nombres largos de distritos, garantizando integridad 100% en las 1,893 llaves `ubigeo` (`zfill(6)`). En IGP Instrumental también se aplicó `format="mixed"` para absorber el cambio de formato horario (`.00` → entero) introducido a partir del `2020-01-01`.

---

## 3. Metodología Geofísica y Matemática

### 3.1. Magnitud de Completitud ($M_c$) y Ley de Gutenberg–Richter

Antes de estimar tasas de sismicidad o parámetros estadísticos, se evalúa el umbral mínimo $M_c$ a partir del cual la red sísmica detecta el 100% de los eventos sin sesgo de registro:

- **Criterio de selección de $M_c$:** Combinación conservadora entre **Máxima Curvatura corregida (`MAXC + 0.2`)** y la **Prueba de Bondad de Ajuste (*Goodness-of-Fit Test*, Wiemer & Wyss 2000)** exigiendo $\text{GFT} \ge 90\%$:
  $$\text{GFT}(M_c) = 100 \cdot \left(1 - \frac{\sum_{M_i \ge M_c} |B_i - S_i|}{\sum_{M_i \ge M_c} B_i}\right)$$
  donde $B_i$ es el conteo acumulado observado y $S_i$ el predicho por la distribución exponencial truncada.
- **Estimación de $b$-value (Máxima Verosimilitud de Aki–Utsu, 1965) e incertidumbre de Shi & Bolt (1982):**
  $$\hat{b} = \frac{\log_{10}(e)}{\bar{M} - (M_c - \Delta M / 2)}, \qquad \hat{\sigma}_b = 2.30 \, \hat{b}^2 \sqrt{\frac{\sum_{i=1}^{N} (M_i - \bar{M})^2}{N(N-1)}}$$
  con $\Delta M = 0.1$.
- **Resultados verificados:**
  - Catálogo instrumental completo (`1960–2026`, 25,878 sismos): $M_c = 4.7$, $N_{\text{comp}} = 12,390$, **$\text{GFT} = 94.75\%$** (máximo `96.29%`), **$b = 1.170 \pm 0.010$**.
  - Época moderna (`2000–2026`, 17,301 sismos): $M_c = 4.7$, $N_{\text{comp}} = 6,746$, **$\text{GFT} = 94.59\%$**, **$b = 1.330 \pm 0.016$**.
  - Evolución histórica de completitud: $M_c$ mejora progresivamente desde `5.1` (`1960–1969`) y `5.5` (`1970–1979`) hasta `4.7` en la era digital moderna.

---

### 3.2. Geometría 3D Wadati–Benioff y Regímenes Sismotectónicos

Las distancias geodésicas a gran escala se calculan sobre la esfera unitaria 3D mediante `scipy.spatial.cKDTree` ($R = 6,371.0\text{ km}$). Sobre `10,330` sismos completos ($M \ge 4.7$) se ajusta una superficie 3D robusta ($L_1$) de la placa de Nazca subductada en una grilla latitudinal (`-4.0°` a `-18.5°`) × distancia a la fosa (`0` a `650 km`):

| Macro-Zona Tectónica | Latitud Representativa | Profundidad de Placa a 300 km | Profundidad de Placa a 500 km | Ángulo de Buzamiento (`250–500 km`) | Error Mediano 3D al Plano |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Norte** (`-3°` a `-9°`) — *Flat-slab* | `-6.5°S` | `79.5 km` | `121.6 km` | **`12.0°`** (sub-horizontal) | `17.6 km` |
| **Centro / Lima** (`-9°` a `-15°`) — *Flat-slab* | `-12.0°S` | `94.3 km` | `130.9 km` | **`11.0°`** (sub-horizontal) | `15.7 km` |
| **Sur** (`-15°` a `-18.5°`) — Subducción normal | `-16.8°S` | `152.5 km` | `268.9 km` | **`30.1°`** (inclinada, arco volcánico) | `15.0 km` |

Todo sismo del catálogo instrumental (`25,878`) se estratifica en **4 regímenes sismotectónicos**:
1. **Subducción Interplaca Costera (`< 60 km`):** `10,400` sismos (`40.2%`).
2. **Losa Intermedia — Wadati–Benioff (`60–350 km`):** `9,932` sismos (`38.4%`).
3. **Falla Cortical Andina / Subandina (`< 60 km` continental):** `5,400` sismos (`20.9%`).
4. **Foco Profundo Amazónico (`> 350 km`, hasta `743 km`):** `146` sismos (`0.6%`, todos al norte de `15°S`).

---

### 3.3. Modelo A — Predicción de Aceleración del Suelo (GMPE Híbrido + Suelo No Lineal)

- **Dataset y objetivo:** `17,794` registros estación–evento (`956` terremotos, `277` estaciones ACELDAT). Variable objetivo: $\log_{10}(\text{PGA}_{H,\text{geom}})$ en Gal ($\text{cm/s}^2$).
- **Protocolo estricto anti-leakage:**
  - Validación cruzada **`GroupKFold(n_splits=5)` agrupada estrictamente por `event_id`** (nunca se divide un mismo terremoto entre entrenamiento y prueba).
  - Evaluación temporal *out-of-time*: entrenamiento en `2019–2024` (`14,718` filas) y prueba en `2025` (`3,076` filas).
  - Los términos empíricos de amplificación de suelo por estación ($\delta S_{2S}$, con regularización Bayesiana *shrinkage* hacia cero en estaciones con pocos registros) se estiman **exclusivamente dentro de cada fold de entrenamiento** (*out-of-fold*).
- **Extrapolación física para megaterremotos ($M_w \ge 8.0$):** Los árboles de decisión saturan por encima del máximo histórico observado en ACELDAT (`Mw 7.5`). Para escenarios extremos se aplica una corrección híbrida usando la pendiente física de magnitud del modelo paramétrico Huber ($c_{1,\text{Huber}} = 0.6201$):
  $$\log_{10}(\text{PGA}_{\text{roca,híbrido}}) = \log_{10}\!\left(\text{PGA}_{\text{LGBM}}(\min(M_w, 7.5), \dots)\right) + 0.6201 \cdot \max(0, M_w - 7.5)$$
- **Amortiguamiento geotécnico no lineal de suelos blandos (tipo Seyhan & Stewart 2014):** Cuando el sacudimiento en roca supera `150 Gal`, los suelos blandos degradan su módulo de corte y disipan energía, evitando picos lineales irreales:
  $$\delta S_{2S,\text{eff}} = \delta S_{2S} \cdot \frac{1}{1 + \left(\dfrac{\text{PGA}_{\text{roca}}}{400\text{ Gal}}\right)^{1.2}}$$
  En sismos débiles/moderados preserva intacta la amplificación observada (ej. Puente Piedra `2.17×` respecto a Morro Solar `1.00×`), mientras que en un megaterremoto Lima $M_w 8.6$ modera las aceleraciones hacia rangos geotécnicamente realistas (`550–610 Gal`).

---

### 3.4. Modelo B — Balance de Momento Sísmico, Monte Carlo y Renovación BPT

El litoral peruano (`-3.5°S` a `-18.5°S`) se divide en **30 segmentos costeros de $0.5^\circ$** y **5 macro-bloques tectónicos**. Para cada tramo:
- **Simulación Monte Carlo ($N = 10,000$ iteraciones):** Propaga la incertidumbre del coeficiente de acoplamiento intersísmico ($\chi$), rigidez elástica ($\mu$), ancho sismogénico ($W$) y velocidad de convergencia ($v$) para estimar el déficit de momento sísmico $M_{0,\text{déficit}}$ y la magnitud potencial equivalente $M_w = \frac{2}{3}\log_{10}(M_0) - 6.07$ en percentiles **P10, P50 y P90**.
- **Probabilidad condicional *Brownian Passage Time* (BPT, Matthews et al. 2002):** Modela el tiempo de espera entre grandes terremotos con media $T_r$ y aperiodicidad $\alpha \in \{0.3, 0.5\}$, calculando la probabilidad condicional de ruptura en los próximos $\Delta T \in \{30, 50\}$ años dado el tiempo transcurrido $t_e$ desde el último megaterremoto:
  $$P(t_e \le T \le t_e + \Delta T \mid T \ge t_e) = \frac{F_{\text{BPT}}(t_e + \Delta T) - F_{\text{BPT}}(t_e)}{1 - F_{\text{BPT}}(t_e)}$$
- **Validación espacial con `HDBSCAN`:** Sobre `2,825` sismos modernos interplaca, identifica `2` grandes clústeres de acoplamiento con solo `5.66%` de ruido y verifica la baja densidad sísmica relativa en los silencios de **Lima–Chincha (1746)** e **Ilo–Tacna (1868/1604)**.

| Macro-Bloque Costero | Último Megaterremoto | Años Acumulados | $M_w$ Potencial P50 (P10–P90) | Prob. BPT 30 años ($\alpha=0.5$ / $0.3$) | Prob. BPT 50 años ($\alpha=0.5$ / $0.3$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Laguna Central: Lima – Callao – Chincha** (`-11.5°` a `-13.5°`) | `1746` | `280 años` | **`8.62`** (`8.54–8.69`) | **`36.5%`** / `63.5%` | **`53.1%`** / `81.7%` |
| **Extremo Sur: Mollendo – Ilo – Tacna** (`-17.0°` a `-18.5°`) | `1868/1869` | `157 años` | **`8.39`** (`8.31–8.47`) | **`34.0%`** / `51.2%` | **`50.5%`** / `71.7%` |
| **Costa Norte: Tumbes – Piura – Chimbote** (`-3.5°` a `-9.0°`) | `1953` | `73 años` | **`8.46`** (`8.37–8.53`) | **`21.5%`** / `12.1%` | **`36.9%`** / `29.3%` |
| **Costa Centro-Norte: Áncash – Huacho** (`-9.0°` a `-11.5°`) | `1966` | `60 años` | **`8.20`** (`8.11–8.28`) | **`16.9%`** / `5.5%` | **`31.5%`** / `18.3%` |
| **Bloque Sur Amplio: Nasca – Arequipa – Tacna** (ruptura 2001) | `2001` | `25 años` | **`8.08`** (`7.93–8.18`) | **`2.9%`** / `0.04%` | **`11.4%`** / `1.3%` |

---

### 3.5. Modelo C — Pronóstico Espacio-Temporal H3 (`res=5`) sin Fuga Temporal

- **Panel causal (`h3_spatiotemporal_panel.parquet`):** `2,650,215` filas (`8,205` celdas H3 × `323` cortes mensuales desde `2000-01-01` hasta `2026-11-01`), con exclusión estricta de ventanas censuradas a la derecha.
- **Validación *Expanding-Window Walk-Forward* (4 splits temporales: `2018–2019`, `2020–2021`, `2022–2023`, `2024–2025`):** Con *purge gap* de **4 meses** para horizontes de 90 días y **12 meses** para horizontes de 365 días, evitando cualquier superposición de etiquetas futuras.
- **Calibración probabilística:** Regresión logística de **Platt** ajustada exclusivamente en la ventana de validación previa al split de prueba, comparando contra dos líneas base geofísicas: **Poisson estacionario** y **ETAS paramétrico** (decaimiento de réplicas tipo Omori–Utsu).
- **Proyecciones derivadas en el Consultor Distrital:** Para ofrecer horizontes ciudadanos homogéneos manteniendo transparencia matemática:
  - **6 meses ($M \ge 4.5$):** $P_{6\text{m}} = 1 - (1 - P_{12\text{m}})^{0.5}$ (derivado de la tasa anual calibrada).
  - **5 años ($M \ge 4.5$):** $P_{5\text{a}} = 1 - (1 - P_{12\text{m}})^{5}$ (bajo tasa estacionaria moderna).
  - **12 meses ($M \ge 5.0$):** $P_{12\text{m}}(M \ge 5.0) = 1 - (1 - P_{90\text{d}}(M \ge 5.0))^{365/90}$ (proyectado desde el modelo calibrado de 90 días).

---

## 4. Tabla Maestra de Métricas de Validación y Limitaciones Científicas

### 4.1. Resumen Consolidado de Desempeño Verificado

| Componente / Modelo | Esquema de Validación | Métrica Principal | Resultado Obtenido | Referencia / Baseline |
| :--- | :--- | :--- | :---: | :---: |
| **Gutenberg–Richter ($M_c = 4.7$)** | Wiemer & Wyss (2000) GFT sobre 12,390 sismos | Bondad de ajuste (`GFT %`) / $b$-value | **`94.75%`** ($b = 1.170 \pm 0.010$) | Umbral estándar $\ge 90\%$ |
| **Modelo A — GMPE (`LightGBM + δS2S OOF`)** | `GroupKFold(n_splits=5)` por `event_id` (956 sismos) | $R^2$ en $\log_{10}(\text{PGA})$ / RMSE / $\tau$ / $\phi$ | **`0.807`** / `0.283` / `0.235` / `0.193` | Huber paramétrico: `0.650` / `0.381` |
| **Modelo A — GMPE Holdout Temporal 2025** | Entrenamiento `2019–2024` → Test `2025` (3,076 filas) | $R^2$ / RMSE ($\log_{10}$) / MAE (Gal) | **`0.781`** / `0.294` / `1.69 Gal` | Huber 2025: `0.616` / `0.389` |
| **Modelo C — H3 90 días ($M \ge 4.5$)** | Walk-Forward 4 splits + Platt (purge 4m) | ROC-AUC / Brier Skill Score (`BSS`) | **`0.853`** (XGB) / **`0.851`** (LGBM) · BSS **`+0.079`** | Poisson: AUC `0.816` · BSS `0.000` |
| **Modelo C — H3 12 meses ($M \ge 4.5$)** | Walk-Forward 4 splits + Platt (purge 12m) | ROC-AUC / Brier Skill Score (`BSS`) | **`0.850`** (XGB) / **`0.849`** (LGBM) · BSS **`+0.203`** | Poisson: AUC `0.816` · ETAS BSS `+0.154` |
| **Modelo C — H3 90 días ($M \ge 5.0$)** | Walk-Forward 4 splits + Platt (purge 4m) | ROC-AUC / Brier Skill Score (`BSS`) | **`0.835`** (XGB) / **`0.832`** (LGBM) · BSS **`+0.021`** | Poisson: AUC `0.711` · BSS `0.000` |
| **Auditoría Territorial vs. FONDES/SIGRID** | Cruce exacto 1,893 distritos (`ubigeo`) | Subestimación nacional / Lima y Callao | **`150 / 426`** (`35.2%`) / **`21 / 50`** (`42.0%`) | Distritos Alto/Muy Alto listados como Bajo/Muy Bajo |

### 4.2. Limitaciones Científicas Explícitas (Transparencia Obligatoria)

1. **Imposibilidad de predicción determinista:** Ningún modelo científico actual puede predecir el día, la hora ni el minuto de un terremoto. Todos los productos de esta plataforma expresan **probabilidades condicionadas** (riesgo estadístico) y **escenarios deterministas de impacto** (*qué pasaría si ocurre*).
2. **Cobertura heterogénea de estaciones ACELDAT:** `223` de los `1,893` distritos cuentan con factor empírico de sitio directo medido por acelerógrafos IGP (`277` estaciones). En los distritos sin estación propia, el simulador aplica un término de sitio neutro ($\delta S_{2S} = 0$, equivalente a condición promedio de referencia) hasta incorporar mediciones locales de $V_{S30}$ o microzonificación CISMID.
3. **Extrapolación en megaterremotos ($M_w \ge 8.0$):** La red ACELDAT 2019–2025 no contiene aún registros de un evento costero $M_w \ge 8.0$; por ello, la respuesta en megaterremotos combina inferencia empírica ML hasta $M_w 7.5$ con física analítica (pendiente Huber + amortiguamiento no lineal Seyhan & Stewart 2014).
4. **Incertidumbre del catálogo histórico pre-instrumental (1552–1959):** Las magnitudes y epicentros coloniales/republicanos provienen de crónicas históricas e isosistas, presentando mayor incertidumbre espacial que la era instrumental moderna (`1960–2026`).
5. **Peligro sísmico vs. riesgo socioeconómico:** El índice sísmico puro cuantifica la **amenaza física** (sacudimiento, laguna costera, tasa sísmica) y los peligros geológicos GEOCATMIN; las columnas reservadas `ext_*` permiten acoplar en futuras extensiones vulnerabilidad estructural (viviendas de adobe/quincha del Censo INEI), exposición poblacional e infraestructura crítica.
