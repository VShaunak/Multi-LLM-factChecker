# Multi-LLM FactChecker & 3D Consensus Evaluator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20external-brightgreen.svg)](https://docs.python.org/3/library/)
[![UI: Google Material Design](https://img.shields.io/badge/UI-Google%20Material%203-4285F4.svg)](#google-like-frontend)

An open-source, zero-dependency localhost engine and 3D visualizer that evaluates claims and queries across **10 free OpenRouter LLMs in parallel**.

Instead of relying on a biased single model or an external "judge LLM", this system applies **transparent mathematical consensus modeling**, computing all **45 unique pairwise comparisons** to determine consensus truth, claim validity, and answer quality.

---

## ✨ Key Features

- **🚀 10 Parallel Fast Free Models**: Queries OpenRouter's live model catalog sorted by lowest recent median latency.
- **🌐 Complete 3D Consensus Graph**: Interactive 3D visualization mapping models as nodes and semantic similarity as connecting edges.
- **📊 10×10 Similarity Heatmap**: Computes all $\binom{10}{2} = 45$ unique pairwise comparisons using lexical and semantic TF-IDF cosine similarity.
- **🛡️ Quota-Safe Preflight Engine**: Tests a single model first to verify credentials and quota limits before dispatching the remaining nine requests.
- **🎨 Google Material 3 / Gemini Interface**: Clean, modern Google-style aesthetic featuring responsive layouts, search suggestion chips, and instant Light/Dark mode toggling.
- **⚡ Zero External Python Dependencies**: Built entirely using Python's standard library (`http.server`, `urllib`, `threading`, `math`) and standard browser Plotly.js.
- **🧪 Instant Local Demo Mode**: Test and explore all 45 comparisons, ranking math, and 3D graphs immediately without requiring an API key or internet access.

---

## 📐 Mathematical Formulation

### 1. Pairwise Similarity Score
For each pair of models $(i, j)$:

$$\text{Pair}(i, j) = \frac{0.35 \cdot T(i, j) + 0.20 \cdot K(i, j) + 0.20 \cdot C(i, j)}{0.75}$$

Where:
- $T(i, j)$: Cosine similarity between complete answers
- $K(i, j)$: Cosine similarity between extracted key points
- $C(i, j)$: Cosine similarity between extracted claims

$$\binom{10}{2} = \frac{10 \times 9}{2} = 45 \text{ unique pair scores}$$

### 2. Agreement Score ($C$)
The overall agreement of model $i$ across all other models:

$$C_i = \frac{1}{N - 1} \sum_{j \neq i} \text{Pair}(i, j)$$

### 3. Final Overall Score ($S$)
Models are ranked transparently using a customizable weighted linear combination:

$$S_i = w_C \cdot C_i + w_Q \cdot Q_i + w_W \cdot W_i + w_V \cdot V_i$$

- **Agreement ($C$)**: Cross-model corroboration $[0, 1]$ (default: $0.25$)
- **Quality ($Q$)**: Syntactic completeness and readability $[0, 1]$ (default: $0.55$)
- **Confidence ($W$)**: Normalized self-confidence $[0, 1]$ (default: $0.05$)
- **Speed ($V$)**: Relative latency score $[0, 1]$ (default: $0.15$)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/VShaunak/Multi-LLM-factChecker.git
cd Multi-LLM-factChecker
```

### 2. Configure OpenRouter API Key (Optional for Demo Mode)
Copy the example environment file and add your free OpenRouter API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
*(Get a free key at [OpenRouter.ai](https://openrouter.ai/keys). If you don't have a key, you can click **"Sample Demo"** in the web interface to test everything locally).*

### 3. Run the Server

#### Windows:
Double-click `run_server.bat` or run:
```powershell
python app.py --port 8020
```

#### macOS / Linux:
```bash
chmod +x run_server.sh
./run_server.sh
# or: python3 app.py --port 8020
```

### 4. Open in Browser
Visit **[http://127.0.0.1:8020](http://127.0.0.1:8020)** in your browser.

---

## 🖥️ User Interface Overview

- **Search Bar & Suggestion Chips**: Rapidly test prompts such as Bigfoot, Quantum Computing, or Financial analysis.
- **Interactive 3D Graph**: Rotate, zoom, and click any node or edge midpoint to inspect underlying cosine metrics.
- **Similarity Heatmap**: 10×10 color-coded matrix showing pairwise agreement across all models.
- **Model Leaderboard**: Ranked model cards displaying breakdown contributions, latency, and extracted claims.
- **Fact Consensus Tab**: Cross-model claim corroboration and numerical consistency auditing.
- **Download & Export**: Export entire benchmark evaluations to JSON or CSV with one click.

---

## 📁 Project Structure

```text
Multi-LLM-factChecker/
├── app.py                # Standalone HTTP server, evaluation engine & math
├── index.html            # Google Material 3 / Gemini frontend
├── sample_questions.txt  # Ready-to-use benchmark prompts
├── PROJECT_MATH.md       # Detailed mathematical proof and formula documentation
├── run_server.bat        # Windows one-click startup script
├── run_server.sh         # Linux/macOS startup script
├── .env.example          # Environment configuration template
├── .gitignore            # Excludes keys and benchmark output runs
└── runs/                 # Persisted benchmark output logs (JSON/CSV)
```

---

## 🔒 Security & Privacy

- Your `OPENROUTER_API_KEY` is loaded locally by the server and is **never** sent to the client browser or committed to GitHub.
- Evaluation runs are stored only on your local machine in the `runs/` folder.
- No external web tracking, telemetry, or cookies.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
