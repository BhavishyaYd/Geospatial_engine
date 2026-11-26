<div align="center">

<!-- ╔══════════════════════════════════════════════════════════════╗ -->
<!--                   ANIMATED WAVE HEADER                         -->
<!-- ╚══════════════════════════════════════════════════════════════╝ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020c18,40:0a2340,80:0c4a6e,100:0369a1&height=220&section=header&text=Geospatial%20Investigation%20Engine&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=AI-Assisted%20Change%20Detection%20%7C%20CSD-Net%20%7C%20RemoteCLIP%20%7C%20Florence-2&descSize=15&descAlignY=60&animation=fadeIn" width="100%"/>

<!-- ╔══════════════════════════════════════════════════════════════╗ -->
<!--                  ANIMATED TYPING TAGLINE                       -->
<!-- ╚══════════════════════════════════════════════════════════════╝ -->

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=19&duration=2800&pause=900&color=38BDF8&center=true&vCenter=true&width=760&lines=AI-Assisted+Satellite+Imagery+Change+Detection+%F0%9F%9B%B0%EF%B8%8F;CSD-Net+Pixel-Level+Change+Prediction+%F0%9F%A7%A0;RemoteCLIP+%2B+Florence-2+Region+Analysis;GeoTIFF+%7C+WGS84+%7C+GeoJSON+Geospatial+Support;React+%2B+FastAPI+%2B+PyTorch+Full-Stack+Platform" alt="Typing SVG"/>

<br/>

<!-- CTA BUTTONS -->
[![GitHub Repo](https://img.shields.io/badge/💻%20Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aryansengar007/Geo-Intel)
[![Project Status](https://img.shields.io/badge/Status-Core%20Development%20Complete-10b981?style=for-the-badge)](https://github.com/aryansengar007/Geo-Intel)

<br/>

<!-- ANIMATED SKILL ICONS -->
<img src="https://skillicons.dev/icons?i=react,python,pytorch,fastapi&theme=dark&perline=4" />

<br/><br/>

![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DL%20Models-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MapLibre](https://img.shields.io/badge/MapLibre-Map%20Render-396CB2?style=flat-square)
![GeoJSON](https://img.shields.io/badge/GeoJSON-WGS84-16a34a?style=flat-square)

</div>

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

> ⚠️ **Project Status:** Core development and stabilization complete. GEI is an **investigation-assistance tool** — its outputs are model predictions and must be independently verified before operational, legal, or policy decisions.

<br/>

## 🛰️ Overview

**Geospatial Investigation Engine (GEI)** is a full-stack AI platform for detecting and analysing changes between two images of the same geographical area captured at different points in time (**T1** and **T2**).

The backend processes the image pair through a **CSD-Net-based change-detection pipeline**, extracts candidate regions, and enriches them with **RemoteCLIP** and **Florence-2** vision-language analysis. Where valid georeferencing metadata is present, candidate regions are converted from pixel coordinates to **WGS84 geographic features** and displayed on an interactive **MapLibre** map.

<br/>

<div align="center">

| 🔍 Detection | 🧠 Analysis | 🗺️ Mapping | 📊 Dashboard |
|:---:|:---:|:---:|:---:|
| CSD-Net pixel-level change | RemoteCLIP + Florence-2 | GeoTIFF → WGS84 GeoJSON | Interactive results UI |

</div>

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## ✨ Key Features

<details>
<summary><strong>🖼️ Image Upload and Investigation Management — click to expand</strong></summary>
<br/>

| Feature | Detail |
|:---|:---|
| Dual image upload | Earlier T1 and later T2 image inputs |
| Input validation | Format, decodability, and dimension compatibility checks |
| AOI association | Optional Area of Interest linking |
| Unique run IDs | Each investigation gets its own isolated run |
| State tracking | Queued · Running · Completed · Failed |
| Frontend persistence | Investigation index stored in browser local storage |

</details>

<details>
<summary><strong>🤖 AI-Based Change Detection — click to expand</strong></summary>
<br/>

| Feature | Detail |
|:---|:---|
| CSD-Net architecture | Siamese encoder-based image-pair analysis |
| Pixel-level prediction | Full-resolution change mask output |
| Probability maps | Uncertainty-aware change probability output |
| Candidate region extraction | Connected changed pixel grouping |
| Region metadata | Bounding boxes · centroids · pixel area |

</details>

<details>
<summary><strong>🧠 Vision-Language Analysis — click to expand</strong></summary>
<br/>

| Model | Purpose |
|:---|:---|
| **RemoteCLIP** | Supplementary temporal and semantic analysis via image-text similarity |
| **Florence-2** | Natural-language descriptions of candidate image regions |

> ⚠️ Similarity scores are embedding similarities, not calibrated probabilities. Generated descriptions may be inaccurate and require visual verification.

</details>

<details>
<summary><strong>🗺️ Geospatial Integration — click to expand</strong></summary>
<br/>

| Feature | Detail |
|:---|:---|
| GeoTIFF metadata inspection | Reads georeferencing transform from uploaded images |
| Coordinate conversion | Pixel-to-geographic for WGS84 and Web Mercator (EPSG:3857 → 4326) |
| GeoJSON generation | Candidate regions as geographic features |
| Interactive map | MapLibre rendering with Geoman AOI drawing tools |
| Graceful fallback | Missing or unsupported georeferencing handled cleanly |

</details>

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🏗️ System Architecture

<br/>

```
┌─────────────────────────────────────────────────────────────┐
│              React + Vite Frontend                          │
│   Upload Workflow · Investigation History · Results         │
│   MapLibre Interactive Map · Geoman AOI Drawing             │
└──────────────────────┬──────────────────────────────────────┘
                       │  multipart upload: T1, T2, AOI (opt)
┌──────────────────────▼──────────────────────────────────────┐
│              FastAPI Backend                                │
│   Input validation · Run ID creation · Job orchestration   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Background Investigation Pipeline                  │
│                                                             │
│  📐 Image Preprocessing                                     │
│      ↓                                                      │
│  🔍 CSD-Net Change Detection                                │
│      → Change mask / probability map                        │
│      → Candidate region extraction                          │
│      ↓                                                      │
│  🧠 RemoteCLIP  ·  Florence-2                               │
│      ↓                                                      │
│  🗺️  Geospatial Processing (when valid metadata exists)     │
│      ↓                                                      │
│  📄 Run Report + Artifact Storage                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│   FastAPI Result Endpoints → React Results Dashboard        │
└─────────────────────────────────────────────────────────────┘
```

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🛠️ Tech Stack

<br/>

<div align="center">

<img src="https://skillicons.dev/icons?i=react,vite,python,pytorch,fastapi&theme=dark" />

<br/><br/>

| Layer | Technology | Purpose |
|:---:|:---|:---|
| 🖥️ **Frontend** | React + Vite + JavaScript | Upload workflow, dashboard, investigation history |
| 🗺️ **Mapping** | MapLibre + Geoman | Interactive map & AOI drawing tools |
| ⚙️ **Backend** | Python + FastAPI | API, validation, background job orchestration |
| 🤖 **Deep Learning** | PyTorch | Model inference for all AI components |
| 🔍 **Change Detection** | CSD-Net | Pixel-level change prediction |
| 🧠 **Vision-Language** | RemoteCLIP | Image-text similarity & region analysis |
| 📝 **Descriptions** | Florence-2 | Natural-language candidate region descriptions |
| 🖼️ **Image Processing** | OpenCV + Pillow | Image loading, transforms & I/O |
| 🌍 **Geospatial** | GeoTIFF metadata + GeoJSON | Coordinate handling & geographic feature output |

</div>

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 📁 Repository Structure

```
GEI/
│
├── 📂 model_engine/                   ← Python backend & AI pipeline
│   ├── 📂 api/
│   │   ├── main.py                    ← FastAPI app entry point
│   │   ├── jobs.py                    ← Background investigation jobs
│   │   ├── geospatial.py              ← GeoTIFF & coordinate conversion
│   │   ├── test_jobs.py               ← Backend unit tests
│   │   └── test_geospatial.py         ← Geospatial unit tests
│   ├── 📂 manual-test-fixtures/
│   │   └── geospatial/                ← Test fixtures for geo processing
│   └── 📂 outputs/
│       └── api_runs/                  ← Per-run investigation artifacts
│
└── 📂 gei-frontend/                   ← React + Vite frontend
    └── 📂 src/
        ├── 📂 components/
        │   ├── dashboard/             ← Results dashboard components
        │   └── investigations/        ← Upload & investigation management
        ├── 📂 pages/                  ← Route-level page components
        └── 📂 services/               ← API client & HTTP layer
```

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 📦 Prerequisites

<div align="center">

| Requirement | Version | Notes |
|:---:|:---:|:---|
| 🐍 Python | 3.13 compatible | For backend & AI pipeline |
| 🟢 Node.js + npm | Latest stable | For React frontend |
| 🔧 Git | Any | Repository management |
| 🤖 Model Checkpoints | — | Required AI assets — verify before running inference |
| 💻 CPU/GPU Memory | — | GPU availability depends on PyTorch build and hardware |

</div>

> ⚠️ Model checkpoint files are **not included** in this repository. Confirm required model assets and configuration before running inference.

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🚀 Setup and Run

### 1. Backend Setup

```powershell
cd model_engine
```

**Activate existing virtual environment:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Or create a new one:**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Install dependencies:**
```powershell
python -m pip install -r requirements.txt
```

**Start the API server:**
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | URL |
|:---|:---|
| 🔗 Backend API | `http://127.0.0.1:8000` |
| 📖 Interactive Docs | `http://127.0.0.1:8000/docs` |

<br/>

### 2. Frontend Setup

```powershell
cd gei-frontend
npm install
npm run dev
```

> To bind to a specific interface: `npm run dev -- --host 127.0.0.1`

<br/>

### 3. Production Build

```powershell
cd gei-frontend
npm run build
npm run preview    # preview the built frontend locally
```

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🔌 API Overview

<details>
<summary><strong>All Endpoints — click to expand</strong></summary>
<br/>

| Method | Endpoint | Purpose |
|:---:|:---|:---|
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/investigations` | Create a new investigation |
| `GET` | `/api/investigations/{run_id}` | Get investigation status |
| `GET` | `/api/investigations/{run_id}/report` | Retrieve investigation report |
| `GET` | `/api/investigations/{run_id}/files` | List run artifacts |
| `GET` | `/api/investigations/{run_id}/inputs/t1` | Retrieve original T1 image |
| `GET` | `/api/investigations/{run_id}/inputs/t2` | Retrieve original T2 image |
| `GET` | `/api/investigations/{run_id}/inputs/{type}/preview` | PNG preview of T1 or T2 |
| `GET` | `/api/investigations/{run_id}/aoi` | Retrieve associated AOI |
| `GET` | `/api/investigations/{run_id}/geospatial` | Retrieve geospatial metadata |
| `GET` | `/api/investigations/{run_id}/regions.geojson` | Candidate region GeoJSON |

</details>

> 📖 For exact request schemas, response structures and error codes — consult the running FastAPI docs at `/docs`.

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 💾 Outputs and Storage

Each run is fully isolated in its own directory:

```
outputs/
└── api_runs/
    └── <run_id>/
        ├── 📂 inputs/
        │   ├── image_t1.png
        │   └── image_t2.png
        ├── 📂 outputs/
        │   ├── change_mask.png
        │   ├── probability_map.png
        │   ├── overlay.png
        │   └── ...
        ├── pipeline_report.json
        ├── aoi.geojson
        ├── aoi_metadata.json
        └── geospatial_metadata.json
```

> ⚠️ The frontend keeps its investigation index in browser local storage under `gei_ai_investigation_runs`. Clearing browser storage removes the visible history index even when backend run data still exists.

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🧪 Testing

**Backend unit tests (33 tests · 0 failures · 0 errors)**

```powershell
cd model_engine
.\.venv\Scripts\python.exe -m unittest discover -v
```

**Frontend lint**

```powershell
cd gei-frontend
npm run lint
```

**Frontend production build**

```powershell
npm run build
```

> ℹ️ Vite may show a bundle-size advisory for chunks exceeding 500 kB — this is a warning, not a failed build.

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## ⚠️ Limitations

| # | Limitation |
|:---:|:---|
| 01 | **Detection ≠ Verification** — CSD-Net identifies visual changes; it does not establish real-world meaning or legality |
| 02 | **Image Alignment** — Equal dimensions do not guarantee pixels represent the same ground location |
| 03 | **False Positives/Negatives** — Shadows, seasons, illumination, and misalignment affect predictions |
| 04 | **Semantic Uncertainty** — RemoteCLIP scores are embedding similarities, not calibrated probabilities |
| 05 | **Description Uncertainty** — Florence-2 descriptions may be inaccurate and require visual verification |
| 06 | **Georeferencing Dependency** — Geographic mapping requires valid, supported GeoTIFF metadata |
| 07 | **CRS Coverage** — Only WGS84 (EPSG:4326) and Web Mercator (EPSG:3857) conversions are implemented |

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🔮 Future Enhancements

| # | Enhancement | Description |
|:---:|:---|:---|
| 01 | 🛰️ Imagery Integration | Connect to satellite providers for automated multi-date acquisition |
| 02 | 📐 Co-registration | More robust image alignment pipeline |
| 03 | 🌍 Extended CRS Support | Additional coordinate reference systems beyond EPSG:4326 and 3857 |
| 04 | 📊 Evaluation Framework | Labelled dataset evaluation — precision, recall, F1, IoU |
| 05 | 🏗️ Change Classification | Improved building, road, and land-use change classification |
| 06 | 📅 Multi-date Tracking | Track changes across more than two time points |
| 07 | ☁️ Cloud Deployment | Scalable background job processing and cloud hosting |
| 08 | 🔐 Auth & Access Control | User authentication, roles, and investigation access controls |
| 09 | 📋 Audit Trails | Enhanced report export and investigation audit history |

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 🛡️ Responsible Use

GEI provides **AI-generated candidate detections** and supplementary interpretations to support human investigation.

Users should:
- Inspect original T1 and T2 imagery directly
- Validate geospatial alignment independently
- Verify model outputs before making operational, legal, or policy decisions

GEI does not independently establish whether a detected change is illegal construction or any other confirmed real-world event.

<br/>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WAVE DIVIDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,4,8,20&height=2" width="100%"/>

<br/>

## 👨‍💻 Author

<div align="center">

### Aryan Sengar

🌍 Gurgaon, India

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aryan-sengar-786b96290/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aryansengar007)

</div>

<br/>

<!-- ╔══════════════════════════════════════════════════════════════╗ -->
<!--                   ANIMATED WAVE FOOTER                         -->
<!-- ╚══════════════════════════════════════════════════════════════╝ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020c18,40:0a2340,80:0c4a6e,100:0369a1&height=130&section=footer&animation=fadeIn" width="100%"/>

<div align="center">

© 2025 **Aryan Sengar** — All Rights Reserved. Unauthorized copying is strictly prohibited.

<br/>

*If you found this project helpful, consider leaving a* ⭐ *— it means a lot!*

</div>
