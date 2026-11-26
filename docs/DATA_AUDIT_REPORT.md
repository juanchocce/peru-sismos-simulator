# DATA AUDIT REPORT — Auditoría Técnica de Fuentes (`base_informativa/`)

> Auditoría generada programáticamente por [`src/ingestion/audit_base_informativa.py`](../src/ingestion/audit_base_informativa.py) sobre copias de solo lectura. Datos crudos extraídos de ZIPs en `data/raw/geocatmin/` y `data/raw/tectonic_plates/`.

---

## 1. Inventario General (`base_informativa/` — 18 archivos)

| Archivo (ruta relativa) | Ext | Tamaño (MB) | Rol en Arquitectura |
| :--- | :--- | ---: | :--- |
| `datos-sismicos_Historico_1471-01-01_1959-12-31.xlsx` | `.xlsx` | 0.0606 | Catálogo histórico IGP (completitud de grandes sismos) |
| `datos-sismicos_Instrumental_1960-01-01_2026-11-22.xlsx` | `.xlsx` | 1.1998 | Catálogo instrumental IGP (silencios sísmicos, $b$-value, cruce GMPE) |
| `IGP_ACELDAT_Peru_2019-2025_Dataset_0.csv` | `.csv` | 1.9030 | Registros acelerométricos IGP (target PGA en 3 componentes para GMPE) |
| `pagina web geocatmin/geocatmin_datos.zip` | `.zip` | 3.5862 | FileGDB INGEMMET: peligros geológicos y zonas críticas multi-riesgo |
| `tectonicplates/tectonicplates-master.zip` | `.zip` | 7.2442 | Modelo global PB2002 (Peter Bird): fosa Nazca-Sudamérica y placas |
| `FONDES_relacion_distritos_nov26.pdf` | `.pdf` | 0.5084 | Relación distrital FONDES/SIGRID por UBIGEO (texto/tablas extraíbles) |
| `Amb-27.pdf` | `.pdf` | 1.5046 | Paper Tavera & Buforn (sismicidad y sismotectónica de Perú) |
| `Microzonificacion-sismica-CISMID-.pdf` | `.pdf` | 2.9055 | Mapa CISMID (1 pág., sin texto seleccionable) |
| `R01_RIESGO_SISMICO_LIMA_A0-ultimo.pdf` | `.pdf` | 5.4988 | Plano A0 riesgo sísmico Lima (1 pág., texto seleccionable parcial) |
| `reporte red acelerometrica 20260825_20261122.pdf` | `.pdf` | 1.4619 | Reporte Red Acelerométrica CENSIS/IGP (10 págs., texto y tablas) |
| `api usgs.txt` | `.txt` | 0.0130 | Documentación API FDSN Event USGS (`earthquake.usgs.gov/fdsnws/event/1/`) |
| `IGP_ACELDAT_Peru_2019-2025_DiccionarioDatos.txt` | `.txt` | 0.0019 | Diccionario oficial ACELDAT (con discrepancias menores vs CSV) |
| `IGP_catalogo_sismicos_desde_1960_ DiccionarioDatos.txt` | `.txt` | 0.0011 | Diccionario oficial IGP (desfasado respecto a cabeceras reales de `.xlsx`) |
| `mp-2025.webp` | `.webp` | 1.1197 | Mapa/infografía (2068 × 2922 px) |
| `red sistemica nacional peru.webp` | `.webp` | 0.3699 | Diagrama/mapa red sísmica (1000 × 1414 px) |
| `Red_CIP.jpg` | `.jpg` | 0.9667 | Mapa estación acelerométrica CIP (1530 × 1980 px) |
| `Red_IGP.jpg` | `.jpg` | 1.0203 | Mapa estación acelerométrica IGP (1530 × 1980 px) |
| `Red_SENCICO.jpg` | `.jpg` | 0.9572 | Mapa estación acelerométrica SENCICO (1530 × 1980 px) |

---

## 2. Catálogos Sísmicos y Acelerométricos (`.xlsx` y `.csv`)

### 2.1 Resumen Ejecutivo de Dimensiones y Fechas UTC

| Dataset | Filas | Cols | Encoding / Sep | Rango Temporal Real (UTC) | Eventos / Estaciones Únicas |
| :--- | ---: | ---: | :--- | :--- | :--- |
| **IGP Histórico** (`.xlsx`, hoja `IGP - Datos sísmicos`) | 1,098 | 8 | openpyxl `str` | `1552-07-02T10:30:00Z` → `1959-12-02T07:02:52Z` | 1,098 filas |
| **IGP Instrumental** (`.xlsx`, hoja `IGP - Datos sísmicos`) | 25,878 | 6 | openpyxl `str` | `1960-01-13T15:40:34Z` → `2026-11-22T05:51:57Z` | 25,878 filas |
| **ACELDAT 2019–2025** (`.csv`) | 18,313 | 23 (13 útiles) | `cp1252` / `;` | `2019-01-03T16:11:48Z` → `2025-12-28T18:19:55Z` | 1,052 timestamps evento, 277 estaciones, 232 UBIGEOs |

### 2.2 Esquemas Reales y Nulos (incluyendo pseudo-nulos `'-'` y espacios)

#### A) `datos-sismicos_Historico_1471-01-01_1959-12-31.xlsx` (1,098 × 8)
| Columna | Dtype Inferido | Nulos / `'-'` | % Faltante | Min | Mediana | Max | Media |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fecha UTC` | `str` | 0 | 0.00% | `1552-07-02` | — | `1959-12-02` | — |
| `hora UTC` | `str` | 0 | 0.00% | — | — | — | — |
| `latitud (º)` | `float64` | 0 | 0.00% | -20.05 | -12.20 | -1.75 | -11.9697 |
| `longitud (º)` | `float64` | 0 | 0.00% | -84.00 | -76.20 | -68.00 | -75.5460 |
| `profundidad (km)` | `str` | 59 (`'-'`) | 5.37% | 5.0 | 40.0 | 700.0 | 57.0048 |
| `magnitud (mb)` | `str` | 123 (`'-'`) | 11.20% | 2.8 | 4.3 | 7.8 | 4.3357 |
| `magnitud (Ms)` | `str` | 894 (`'-'`) | 81.42% | 4.5 | 6.5 | 9.4 | 6.6363 |
| `magnitud (Mw)` | `str` | 1,086 (`'-'`) | 98.91% | 6.8 | 8.1 | 8.8 | 8.0500 |

#### B) `datos-sismicos_Instrumental_1960-01-01_2026-11-22.xlsx` (25,878 × 6)
| Columna | Dtype Inferido | Nulos / `'-'` | % Faltante | Min | P25 | Mediana | P75 | Max | Media |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fecha UTC` | `str` | 0 | 0.00% | `1960-01-13` | — | — | — | `2026-11-22` | — |
| `hora UTC` | `str` | 0 | 0.00% | — | — | — | — | — | — |
| `latitud (º)` | `float64` | 0 | 0.00% | -20.3100 | -15.3910 | -11.7992 | -7.9700 | -1.0700 | -11.3949 |
| `longitud (º)` | `float64` | 0 | 0.00% | -82.8937 | -77.7300 | -75.7300 | -73.6066 | -67.6400 | -75.6925 |
| `profundidad (km)` | `int64` | 0 | 0.00% | 5.0 | 29.0 | 48.0 | 105.0 | 743.0 | 70.2653 |
| `magnitud (M)` | `float64` | 0 | 0.00% | 3.0 | 4.5 | 4.6 | 4.9 | 8.4 | 4.6660 |

#### C) `IGP_ACELDAT_Peru_2019-2025_Dataset_0.csv` (18,313 × 23; 13 columnas útiles + 10 `Unnamed: 13..22` 100% nulas)
| Columna | Dtype Inferido | % Nulos | Min / Ejemplo | P25 | Mediana | P75 | Max | Media |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FECHA_CORTE` | `int64` | 0.00% | `20251231` (formato aaaammdd) | — | — | — | — | — |
| `UBIGEO` | `int64` | 0.00% | `21801` (5 dígitos) / `250105` | — | — | — | — | — |
| `DEPARTAMENTO` | `str` | 0.00% | texto `cp1252` | — | — | — | — | — |
| `PROVINCIA` | `str` | 0.00% | texto `cp1252` | — | — | — | — | — |
| `DISTRITO` | `str` | 0.00% | texto `cp1252` | — | — | — | — | — |
| `FECHA_EVENTO` | `str` | 0.00% | `20190103` y `4/18/2025` | — | — | — | — | — |
| `HORA_EVENTO` | `int64` | 0.00% | `5228` (4 díg.) / `161148` | — | — | — | — | — |
| `COD_ESTACION` | `str` | 0.00% | 277 códigos únicos | — | — | — | — | — |
| `LAT_ESTACION` | `float64` | 0.00% | -18.1582 | -14.0890 | -12.1030 | -11.4860 | -3.5234 | -12.3074 |
| `LON_ESTACION` | `float64` | 0.00% | -81.2730 | -77.0720 | -76.8260 | -74.6680 | -69.1830 | -75.9320 |
| `ACEL_MAX_VERTICAL` | `float64` | 0.00% | 0.0039 | 0.1785 | 0.4370 | 1.3491 | 245.5761 | 2.2615 |
| `ACEL_MAX_NORTE-SUR` | `float64` | 0.00% | 0.0047 | 0.2703 | 0.6531 | 2.0450 | 320.8798 | 3.5388 |
| `ACEL_MAX_ESTE-OESTE` | `float64` | 0.00% | 0.0028 | 0.2711 | 0.6619 | 2.0820 | 387.2492 | 3.5839 |
| `Unnamed: 13..22` | `float64` | 100.00% | 10 cols vacías por `;` finales | — | — | — | — | — |

### 2.3 Anomalías Críticas Detectadas en Tabulares

| Dataset | Columna / Aspecto | Hallazgo Cuantificado | Mitigación Obligatoria en Ingesta |
| :--- | :--- | :--- | :--- |
| **ACELDAT** | `UBIGEO` | **5,131 / 18,313 filas (28.02%)** tienen longitud 5 (`21801`, `50101`, `70105`); pérdida de cero inicial. | Leer como `str` + `.str.strip().str.zfill(6)`. |
| **ACELDAT** | `HORA_EVENTO` | **7,195 / 18,313 filas (39.29%)** tienen longitud < 6 (`2`: 4, `3`: 312, `4`: 611, `5`: 6,268; ej. `34950`, `5228`). | Leer como `str` + `.str.strip().str.zfill(6)`. |
| **ACELDAT** | `FECHA_EVENTO` | Formato mixto: 17,042 filas `YYYYMMDD` (len 8) y **1,271 filas (6.94%)** en 2025 con formato `M/D/YYYY` (len 9–10, ej. `4/18/2025`). | Parser dual (`%Y%m%d` + `%m/%d/%Y`), jamás `pd.to_datetime` sin especificar ambos. |
| **ACELDAT** | Encoding & Delimitador | `cp1252` (contiene `Ñ` en byte `0xD1`; falla `utf-8`) y 10 `;` extra al final de cada fila (`Unnamed: 13..22`). | Leer con `encoding="cp1252", sep=";"` y filtrar `~col.startswith("Unnamed")`. |
| **ACELDAT** | Sin `event_id` ni parámetros de fuente | No trae lat/lon/profundidad/Mw del sismo; solo `FECHA_EVENTO`, `HORA_EVENTO` y ubicación de estación. | Cruzar por ventana temporal UTC ($\pm\Delta t$) contra IGP Instrumental / USGS para asignar `event_id` y agrupar en `GroupKFold`. |
| **IGP Instrumental** | `hora UTC` formato mixto | Hasta `2019-12-31` usa `HH:MM:SS.00` (21,068 filas); desde `2020-01-01` usa `HH:MM:SS` sin decimales (4,810 filas). | Usar `pd.to_datetime(..., format="mixed", utc=True)`. |
| **IGP Histórico / Inst.** | Desfase vs `DiccionarioDatos.txt` | Diccionario lista `FECHA_CORTE`, `FECHA_UTC`, `HORA_UTC`, `LATITUD`, `LONGITUD`, `PROFUNDIDAD`, `MAGNITUD`; archivos reales usan minúsculas con unidades (`fecha UTC`, `latitud (º)`, `magnitud (M)` / `mb`/`Ms`/`Mw`). | Mapear explícitamente desde los nombres reales del `.xlsx`. |
| **IGP Histórico** | Fechas pre-1677 | Primer registro real es `1552-07-02` (no `1471`). | Usar resolución `datetime64[us, UTC]` o `datetime64[s, UTC]` (evitar desbordamiento de `datetime64[ns]`). |

---

## 3. Archivos ZIP Geoespaciales (`geocatmin` y `tectonicplates`)

### 3.1 `pagina web geocatmin/geocatmin_datos.zip` → `data/raw/geocatmin/`
Contiene **53 archivos internos** formando una File Geodatabase ESRI (`zipfolder/data.gdb`, driver `OpenFileGDB`, CRS `EPSG:4326`).

| Capa en `data.gdb` | Geometría | Features | Bounds (`EPSG:4326`) | Columnas Clave / Hallazgos | Utilidad para el Modelado |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **`Peligros_Geologicos`** | `Point` | 37,023 | `[-81.307, -18.299, -68.798, -1.560]` | 39 cols. `TIPO_PELIG` (Flujo 9,725; Caída 9,398; Otro Peligro 8,817; Deslizamiento 7,324; Mov. Complejo 912; Reptación 789; Vuelco 50; Volcánico 3; Desplaz. Lateral 1; NaN 4), `PELIGRO_ES`, `GRADO_PELI`, `GRADO_VULN`, `PENDIENTE`, `CD_DIST` (35,687 de 6 díg.; 1,336 NaN), `FECHA_MUES` (35,334 NaN). Espacios `' '` usados como pseudo-nulos. | **Capa principal multi-riesgo:** susceptibilidad de remoción en masa co-sísmica y peligros geológicos indexable por `CD_DIST` (UBIGEO 6 díg.) y hexágonos H3. |
| **`Zonas_Criticas`** | `Point` | 2,177 | `[-81.305, -18.067, -68.963, -2.487]` | 29 cols. `PELIGROS_G` (Deslizamiento 241, Flujo detritos 214, Inundación fluvial 108, Derrumbe 81, Erosión fluvial 78...), `CD_DIST` (2,177/2,177 de 6 díg., 0 nulos), `ELEMENTO`, `INDICADORACTIVO`. `FOTO` y `ANIO` 100% nulos. | **Priorización territorial:** inventario validado de zonas críticas activas para riesgo compuesto por distrito/H3. |
| **`Limite_Paises`** | `MultiPolygon` | 1 | `[-81.355, -18.349, -68.674, -0.037]` | `PLACENAME="Peru"`, `SHAPE_Length`, `SHAPE_Area`. | Máscara territorial de Perú para recorte espacial y grillado H3. |

### 3.2 `tectonicplates/tectonicplates-master.zip` → `data/raw/tectonic_plates/`
Contiene **50 archivos internos** (GeoJSON, ESRI Shapefiles, `original/` ASCII, `PB2002_poles.xls`, `README.md`, `LICENSE.md`, `example_plates.png`). CRS uniforme: `EPSG:4326`.

| Archivo exacto dentro del ZIP (`tectonicplates-master/...`) | Driver | Geom | Features | Subconjunto Exacto para Perú y Utilidad en Modelado |
| :--- | :--- | :--- | ---: | :--- |
| **`GeoJSON/PB2002_boundaries.json`** | `GeoJSON` | `LineString` | 241 | **Fosa / Límite de placa:** filtrar `Name == "NZ\\SA"` (2 `LineString`, `Type="subduction"`). Nota: usa `\` (no `-`) indicando que Nazca (`NZ`) subducta bajo Sudamérica (`SA`). Sirve para distancia a la fosa (`dist_trench_km`). |
| **`GeoJSON/PB2002_steps.json`** | `GeoJSON` | `LineString` | 5,824 | **Features geofísicas por segmento:** filtrar `PLATEBOUND == "NZ\\SA"` (60 pasos) + `"NZ\\ND"` (16) + `"NZ\\AP"` (12). Provee `VELOCITYLE` (tasa de convergencia), `VELOCITYAZ`, `SEAFLOORAG` (edad de placa oceánica), `ELEVATION` y `STEPCLASS`. |
| **`GeoJSON/PB2002_plates.json`** | `GeoJSON` | `Polygon`/`MultiPolygon` | 54 | **Polígonos de placa:** filtrar `Code.isin(["NZ", "SA"])` para asignar pertenencia de placa a eventos. |
| **`GeoJSON/PB2002_orogens.json`** | `GeoJSON` | `Polygon`/`MultiPolygon` | 13 | **Orógenos:** filtrar `Name == "Peru"` para identificar dominio cortical andino / deformación difusa. |

> *Nota:* Los `.shp` en la raíz de `tectonicplates-master/` tienen exactamente las mismas geometrías y conteos que los `.json` en `GeoJSON/`, pero los `.geojson` evitan truncamiento de nombres de campos y tienen 0 nulos en `Type` de `PB2002_boundaries.json` (vs 176 nulos en `.shp`). Usar la carpeta `GeoJSON/`.

---

## 4. Documentos PDF e Imágenes

### 4.1 PDFs (verificados con `pdfplumber`)

| Archivo | Tamaño (MB) | Páginas | Texto Seleccionable | Tablas (muestra 3 págs.) | Hallazgo / Uso |
| :--- | ---: | ---: | :--- | ---: | :--- |
| **`FONDES_relacion_distritos_nov26.pdf`** | 0.5084 | 32 | **Sí (32/32 págs., 154,159 chars)** | 3 tablas detectadas | **Totalmente extraíble con `pdfplumber`:** contiene `UBIGEO` de 6 dígitos (ej. `010101`), `DEPARTAMENTO`, `PROVINCIA`, `DISTRITO` y `NIVEL DE PELIGRO Y/O RIESGO SIGRID`. Listo para parser tabular estructurado. |
| `reporte red acelerometrica 20260825_20261122.pdf` | 1.4619 | 10 | Sí (4,028 chars) | 3 | Reporte IGP/ACELDAT de ejemplo (sismo `22/11/2026`, M5, 145 km prof., Junín). Útil para validar formato de reportes CENSIS. |
| `Amb-27.pdf` | 1.5046 | 34 | Sí (4,825 chars) | 0 | Marco teórico Tavera & Buforn (regímenes sísmicos de Perú y transición ~15°S). |
| `R01_RIESGO_SISMICO_LIMA_A0-ultimo.pdf` | 5.4988 | 1 | Sí (3,315 chars) | 2 | Cartografía A0 de riesgo sísmico en Lima Metropolitana (coordenadas UTM detectables). |
| `Microzonificacion-sismica-CISMID-.pdf` | 2.9055 | 1 | **No (0 chars)** | 0 | Mapa rasterizado/curvas sin capa de texto; requiere georreferenciación manual si se usa. |

### 4.2 Imágenes (referencia visual)

| Archivo | Formato | Modo | Resolución (px) | Tamaño (MB) |
| :--- | :--- | :--- | :---: | ---: |
| `mp-2025.webp` | WEBP | RGB | 2068 × 2922 | 1.1197 |
| `red sistemica nacional peru.webp` | WEBP | RGB | 1000 × 1414 | 0.3699 |
| `Red_CIP.jpg` | JPEG | RGB | 1530 × 1980 | 0.9667 |
| `Red_IGP.jpg` | JPEG | RGB | 1530 × 1980 | 1.0203 |
| `Red_SENCICO.jpg` | JPEG | RGB | 1530 × 1980 | 0.9572 |
