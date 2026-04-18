# Multi-Model Distribution Forecast Pipeline

Simultaneously sends **100 independent forecast prompts per firm** to **3 LLM providers** (OpenAI, Anthropic Claude, Google Gemini), then computes forecast distributions.

---

## Architecture

```
Professor's API (yowlo.app)
         │
         ▼
   Normalize firms
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
      OpenAI            Anthropic           Gemini
   100 draws/firm     100 draws/firm     100 draws/firm
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                   Aggregate all draws
                            │
                            ▼
              Distribution statistics per firm
             (per-model + combined all models)
                            │
                            ▼
               CSV / JSON + Google Drive
```

For **F firms**: total API calls = **F × 3 × 100 = F × 300**

All calls run concurrently via `asyncio` + `aiohttp` with per-provider semaphores.

---

## Package contents

| File | Purpose |
|---|---|
| `forecast_distribution_pipeline.py` | Main pipeline |
| `.env.example` | Configuration template (copy to `.env`) |
| `requirements.txt` | Python dependencies |
| `run_forecast_now.bat` | Quick run helper (Windows) |
| `register_windows_task.ps1` | Schedule daily runs (Windows) |
| `credentials.json` | (you provide) Google Drive OAuth |
| `token.json` | (auto-created) Drive refresh token |

---

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Create `.env`

```bash
cp .env.example .env
```

Fill in your three API keys:

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

The professor's data API settings are already pre-filled from your existing config.

### 3) Google Drive (same as before)

If you already have `credentials.json` and `token.json` from the original pipeline, just copy them into this folder. Otherwise:

```bash
python forecast_distribution_pipeline.py --authorize-drive-only
```

---

## Usage

### Quick test (5 draws, no Drive upload)

```bash
python forecast_distribution_pipeline.py --n-draws 5 --skip-drive-upload
```

### Full run (100 draws × 3 models × all firms)

```bash
python forecast_distribution_pipeline.py
```

### Run with custom draw count

```bash
python forecast_distribution_pipeline.py --n-draws 50
```

### Skip Drive upload

```bash
python forecast_distribution_pipeline.py --skip-drive-upload
```

---

## Output files (in `output/<run_id>/`)

| File | Description |
|---|---|
| `per_draw_results.csv` | Every single draw: firm × model × draw# with forecast_eps, forecast_sales, confidence, reasoning |
| `distribution_summary.csv` | Stats per firm × model: mean, median, std, p10, p25, p75, p90 for EPS and sales |
| `distribution_summary.json` | Same stats in JSON format |
| `failures.json` | Any failed API calls with error details |
| `run_summary.json` | Run metadata (counts, providers, timing) |
| `normalized_records.json` | Parsed company data from the API |
| `raw_api_payload.json` | Raw API response (if SOURCE_API_SAVE_RAW=true) |
| `pipeline.log` | Full execution log |

### `distribution_summary.csv` structure

Each firm gets **4 rows**: one for each model + one `combined` row that pools all 300 draws:

| company | model_provider | n_draws_ok | eps_mean | eps_median | eps_std | eps_p10 | eps_p90 | sales_mean | ... |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | openai | 100 | 1.52 | 1.51 | 0.08 | 1.41 | 1.63 | 95000 | ... |
| AAPL | anthropic | 100 | 1.48 | 1.47 | 0.10 | 1.35 | 1.61 | 94500 | ... |
| AAPL | gemini | 100 | 1.55 | 1.54 | 0.07 | 1.45 | 1.64 | 96000 | ... |
| AAPL | combined | 300 | 1.52 | 1.51 | 0.09 | 1.40 | 1.63 | 95200 | ... |

---

## Configuration reference

### Draw & temperature

| Variable | Default | Notes |
|---|---|---|
| `N_DRAWS` | 100 | Prompts per firm per model |
| `*_TEMPERATURE` | 0.7 | Higher = wider distribution. 0 = near-deterministic |

### Concurrency (rate-limit protection)

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_CONCURRENCY` | 50 | Max simultaneous OpenAI requests |
| `ANTHROPIC_CONCURRENCY` | 40 | Max simultaneous Anthropic requests |
| `GEMINI_CONCURRENCY` | 50 | Max simultaneous Gemini requests |

Lower these if you hit rate limits. The pipeline has automatic retry with exponential backoff for 429 errors.

### Enable/disable providers

```env
ENABLE_OPENAI=true
ENABLE_ANTHROPIC=true
ENABLE_GEMINI=false    # disable one provider
```

---

## Differences from original pipeline

| Original | New |
|---|---|
| 1 forecast per firm | 100 forecasts per firm |
| OpenAI only | OpenAI + Anthropic + Gemini |
| Sequential or Batch API | Fully async concurrent |
| Single point estimate | Distribution (mean, median, std, percentiles) |
| `forecasts.csv` | `per_draw_results.csv` + `distribution_summary.csv` |
