#!/usr/bin/env python3
"""
Multi-Model Distribution Forecast Pipeline (Agent-Based)
=========================================================
Author: Rui Sun

Uses FIXED analyst personas per provider (generated once by train_agents.py)
to produce forecasts, instead of N anonymous temperature draws.

For each firm × each of 100 personas (per provider) → one forecast.
Distribution per (firm, provider) and a combined-across-providers distribution.

Personas are loaded from agent_pipeline/agents/*.json and never modified at runtime.
Temperature is not set explicitly — each provider uses its own default.

Prompt caching: Anthropic uses explicit cache_control on the system blocks
(task and persona blocks separately). OpenAI / Gemini auto-cache long shared
prefixes — no extra wiring needed.
"""
from __future__ import annotations
__author__ = "Rui Sun"
import argparse, asyncio, io, json, logging, os, sys, zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
# Worktree root may not have .env; walk up from CWD if needed. override=True
# protects against pre-existing empty placeholders in the shell environment.
if (PARENT_DIR / ".env").exists():
    load_dotenv(PARENT_DIR / ".env", override=True)
else:
    load_dotenv(override=True)

from forecast_distribution_pipeline import (
    load_settings, normalize_records,
    parse_json_from_text, parse_payload_from_bytes, render_template_in_obj,
    setup_logging, infer_next_quarter_label,
    compute_data_hash, find_previous_run_hash, compute_distribution,
    save_csv, save_json, authorize_drive,
    ensure_drive_folder, upload_file_to_drive,
    build_history_block, sanitize_text,
    DIST_SUMMARY_FIELDS,
)

AGENTS_DIR = SCRIPT_DIR / "agents"


def _upload_dir_recursive(service, local_dir, drive_parent_id):
    """Mirror a local directory tree into Drive under drive_parent_id."""
    uploaded = []
    for p in sorted(local_dir.iterdir()):
        if p.is_file():
            uploaded.append(upload_file_to_drive(service, p, drive_parent_id))
        elif p.is_dir():
            sub_id = ensure_drive_folder(service, p.name, drive_parent_id)
            uploaded.extend(_upload_dir_recursive(service, p, sub_id))
    return uploaded


def upload_run_folder_to_drive_agents(service, settings, run_dir, run_dt):
    """Upload to nightly_forecasts/<year>agent/<month>/<run_id>/, keeping agent-pipeline
    outputs in a sibling folder ("2026agent") to the original pipeline's "2026".
    Recursive — subfolders like benchmark/ are mirrored into Drive."""
    root_id = settings.google_drive_root_folder_id.strip()
    if not root_id:
        root_id = ensure_drive_folder(service, settings.google_drive_root_folder_name)
    yid = ensure_drive_folder(service, f"{run_dt.year}agent", root_id)
    mid = ensure_drive_folder(service, f"{run_dt.month:02d}", yid)
    rid = ensure_drive_folder(service, run_dir.name, mid)
    uploaded = _upload_dir_recursive(service, run_dir, rid)
    return {"drive_root_folder_id": root_id, "drive_run_folder_id": rid,
            "uploaded_files": uploaded}


def fetch_and_save_api_data(settings, run_dt, run_dir):
    """Hit the data API once. If the response is a ZIP, save ALL members raw to
    run_dir/raw_<name> so the benchmark/estimates file is preserved alongside the
    standardized data. Returns the parsed payload from the preferred ZIP member.

    Replaces the parent module's fetch_source_payload_sync — that one only reads
    the preferred member and discards the others."""
    retry = Retry(total=5, backoff_factor=1.2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    ctx = {"run_date": run_dt.strftime("%Y-%m-%d"),
           "run_datetime_iso": run_dt.isoformat(),
           "forecast_period": settings.forecast_period or infer_next_quarter_label(run_dt)}
    hdrs = render_template_in_obj(settings.data_api_headers, ctx)
    prms = render_template_in_obj(settings.data_api_params, ctx)
    bdy = render_template_in_obj(settings.data_api_body, ctx)

    logging.info("Fetching from API: %s", settings.data_api_url)
    if settings.data_api_method == "GET":
        resp = session.get(settings.data_api_url, headers=hdrs, params=prms,
                           timeout=settings.data_api_timeout_seconds)
    elif settings.data_api_method == "POST":
        resp = session.post(settings.data_api_url, headers=hdrs, params=prms,
                            json=bdy, timeout=settings.data_api_timeout_seconds)
    else:
        raise ValueError("Method must be GET or POST.")
    if resp.status_code >= 400:
        raise RuntimeError(f"API failed: {resp.status_code}: {resp.text[:500]}")

    ct = resp.headers.get("Content-Type", "").lower()
    cd = resp.headers.get("Content-Disposition", "").lower()

    if "application/zip" in ct or "x-zip" in ct or ".zip" in cd or resp.content[:2] == b"PK":
        preferred = os.getenv("DATA_API_ZIP_MEMBER", "").strip()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            logging.info("ZIP members: %s", members)
            for m in members:
                name = Path(m).name
                nl = name.lower()
                # Put consensus / benchmark files into their own subfolder, matching the
                # original pipeline's layout (<run_id>/benchmark/data_estimates.csv).
                if "estimate" in nl or "benchmark" in nl or "consensus" in nl:
                    target = run_dir / "benchmark" / name
                else:
                    target = run_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(m))
                logging.info("Saved API member: %s (%d bytes)",
                             target.relative_to(run_dir), target.stat().st_size)
            if preferred:
                member = None
                for f in members:
                    fn = Path(f).stem.lower()
                    ff = Path(f).name.lower()
                    pl = preferred.lower()
                    if ff == pl or fn == pl or pl in fn:
                        member = f
                        break
                if not member:
                    member = members[0]
                    logging.warning("ZIP member %s not found, using %s", preferred, member)
            else:
                member = sorted(members, key=lambda n: (
                    0 if "standardized" in n.lower() else 1,
                    1 if "benchmark" in n.lower() else 0,
                    n.lower()))[0]
            logging.info("Using ZIP member for parsing: %s", member)
            return parse_payload_from_bytes(zf.read(member), filename=member)

    # Non-ZIP: save raw response then parse
    (run_dir / "raw_api_response.bin").write_bytes(resp.content)
    try:
        return resp.json()
    except json.JSONDecodeError:
        return parse_payload_from_bytes(resp.content,
                                        filename=settings.data_api_url, content_type=ct)


DRAW_OUTPUT_FIELDS = [
    "run_id", "run_date_pt", "forecast_period",
    "company_id", "ticker", "company_name", "currency",
    "model_provider", "model_name", "agent_id", "agent_name", "draw_number",
    "forecast_eps", "forecast_sales", "forecast_profit_margin", "confidence",
    "reasoning",
    "status", "error_message",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cache_creation_tokens", "cache_read_tokens",
]

TASK_SYSTEM_BLOCK = (
    "Your task: forecast the next period for exactly one company using ONLY that company's "
    "provided history. Apply your methodology and bias naturally — do not exaggerate or "
    "caricaturize your role. Do not use other companies, sector averages, or external knowledge "
    "beyond what is implicit in your role. If data are sparse, stay conservative and lean on "
    "the most recent observations.\n\n"
    "Return ONLY valid JSON with these exact keys: "
    "company_id (string), ticker (string), company_name (string), forecast_period (string), "
    "forecast_eps (number), forecast_sales (number), "
    "forecast_profit_margin (NET profit margin = net income / total revenue, as decimal — "
    "e.g. 0.15 means 15%; this is NOT gross margin and NOT operating margin), "
    "confidence (number between 0 and 1), "
    "reasoning (string, 1-2 concise sentences naming: (i) which forecasting method/tool you "
    "used — e.g. DCF, multiples, trend extrapolation, channel-check, momentum, scenario-weighted; "
    "and (ii) the single most decisive piece of evidence or assumption from the firm's history "
    "that drove your forecast values).\n"
    "No markdown, no code fences, no preamble — just the raw JSON object."
)

# Anthropic-specific addendum: Haiku has a persistent ~7% prior on net profit margin
# regardless of clarifying definitions. Append a targeted calibration nudge ONLY for
# Anthropic calls so OpenAI / Gemini outputs stay un-anchored.
ANTHROPIC_CALIBRATION_NOTE = (
    "Anthropic-specific calibration: When estimating forecast_profit_margin, do NOT default "
    "to low values around 5-10% out of a generic 'profit margin = small number' prior. "
    "For mature firms with consistent positive operating performance and stable revenue, "
    "NET profit margin is typically 12-25%, even though gross margin can be 50-70%. "
    "Cross-check by computing (forecast_eps × diluted_shares_outstanding) / forecast_sales — "
    "if this implies a share count that's wildly inconsistent with the firm's actual size, "
    "revise your margin upward."
)


def load_personas(enabled_providers):
    personas = {}
    for provider in enabled_providers:
        path = AGENTS_DIR / f"{provider}_agents.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run train_agents.py first to generate personas."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path} must be a non-empty JSON array")
        personas[provider] = data
        logging.info("Loaded %d %s personas from %s", len(data), provider, path.name)
    return personas


def render_persona_block(persona):
    name = persona.get("name", "an equity analyst")
    years = persona.get("years_experience", 10)
    sectors = ", ".join(persona.get("sector_focus", []) or ["generalist"])
    method = persona.get("methodology", "fundamental analysis")
    bias = persona.get("optimism_bias", "balanced")
    guidance = persona.get("guidance_weight", "neutral")
    risk = persona.get("risk_tolerance", "moderate")
    beliefs = persona.get("notable_beliefs", "")
    lines = [
        f"You are \"{name}\", a sell-side equity research analyst with {years} years of experience.",
        f"Sector focus: {sectors}.",
        f"Forecasting methodology: {method}",
        f"Optimism bias: {bias}. Guidance weight: {guidance}. Risk tolerance: {risk}.",
    ]
    if beliefs:
        lines.append(f"Notable beliefs: {beliefs}")
    return "\n".join(lines)


def build_user_prompt(record, forecast_period, settings):
    hblock, _ = build_history_block(
        record,
        max_points=settings.prompt_max_history_points,
        note_chars=settings.prompt_max_note_chars,
        include_eps=settings.prompt_include_eps_history,
    )
    ctx = sanitize_text(record.get("context", ""), settings.prompt_max_company_context_chars)
    return (
        f"Company ID: {record.get('company_id','')}\n"
        f"Ticker: {record.get('ticker','')}\n"
        f"Company name: {record.get('company_name','')}\n"
        f"Currency: {record.get('currency','')}\n"
        f"Forecast period: {forecast_period}\n\n"
        f"Recent financial history:\n{hblock}\n\n"
        f"Additional context: {ctx or 'None'}\n\nReturn the JSON forecast now."
    )


async def call_openai(aio, sem, settings, task_block, persona_block, user_prompt):
    async with sem:
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": task_block + "\n\n" + persona_block},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": settings.openai_max_completion_tokens,
        }
        headers = {"Authorization": f"Bearer {settings.openai_api_key}",
                   "Content-Type": "application/json"}
        for attempt in range(5):
            async with aio.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 429 or 500 <= resp.status < 600:
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
                    continue
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"OpenAI {resp.status}: {json.dumps(body)[:400]}")
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                parsed = parse_json_from_text(text)
                return {
                    "forecast_eps": parsed.get("forecast_eps"),
                    "forecast_sales": parsed.get("forecast_sales"),
                    "forecast_profit_margin": parsed.get("forecast_profit_margin"),
                    "confidence": parsed.get("confidence"),
                    "reasoning": str(parsed.get("reasoning", "")),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": cached,
                }
        raise RuntimeError("OpenAI: max retries (429)")


async def call_anthropic(aio, sem, settings, task_block, persona_block, user_prompt):
    # Anthropic-specific calibration nudge to combat persistent low-PM anchor
    task_block = task_block + "\n\n" + ANTHROPIC_CALIBRATION_NOTE
    async with sem:
        payload = {
            "model": settings.anthropic_model,
            "max_tokens": settings.anthropic_max_tokens,
            "system": [
                {"type": "text", "text": task_block, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": persona_block, "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {"x-api-key": settings.anthropic_api_key,
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        for attempt in range(5):
            async with aio.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 429 or 500 <= resp.status < 600:
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
                    continue
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"Anthropic {resp.status}: {json.dumps(body)[:400]}")
                text = "".join(b.get("text", "") for b in body.get("content", [])
                               if b.get("type") == "text")
                usage = body.get("usage", {})
                parsed = parse_json_from_text(text)
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                return {
                    "forecast_eps": parsed.get("forecast_eps"),
                    "forecast_sales": parsed.get("forecast_sales"),
                    "forecast_profit_margin": parsed.get("forecast_profit_margin"),
                    "confidence": parsed.get("confidence"),
                    "reasoning": str(parsed.get("reasoning", "")),
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                }
        raise RuntimeError("Anthropic: max retries (429)")


async def call_gemini(aio, sem, settings, task_block, persona_block, user_prompt):
    async with sem:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}")
        payload = {
            "system_instruction": {"parts": [{"text": task_block + "\n\n" + persona_block}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": settings.gemini_max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        for attempt in range(5):
            async with aio.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 429 or 500 <= resp.status < 600:
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
                    continue
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"Gemini {resp.status}: {json.dumps(body)[:400]}")
                text = "".join(p.get("text", "") for c in body.get("candidates", [])
                               for p in c.get("content", {}).get("parts", []))
                um = body.get("usageMetadata", {})
                parsed = parse_json_from_text(text)
                return {
                    "forecast_eps": parsed.get("forecast_eps"),
                    "forecast_sales": parsed.get("forecast_sales"),
                    "forecast_profit_margin": parsed.get("forecast_profit_margin"),
                    "confidence": parsed.get("confidence"),
                    "reasoning": str(parsed.get("reasoning", "")),
                    "prompt_tokens": um.get("promptTokenCount", 0),
                    "completion_tokens": um.get("candidatesTokenCount", 0),
                    "total_tokens": um.get("totalTokenCount", 0),
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": um.get("cachedContentTokenCount", 0),
                }
        raise RuntimeError("Gemini: max retries (429)")


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}


async def run_agent_forecast(aio, sem, settings, provider, model_name, persona,
                             record, forecast_period, draw_number, run_id, run_date_pt):
    persona_block = render_persona_block(persona)
    user_prompt = build_user_prompt(record, forecast_period, settings)
    base = {
        "run_id": run_id, "run_date_pt": run_date_pt, "forecast_period": forecast_period,
        "company_id": record["company_id"], "ticker": record.get("ticker", ""),
        "company_name": record.get("company_name", ""), "currency": record.get("currency", ""),
        "model_provider": provider, "model_name": model_name,
        "agent_id": persona.get("agent_id", ""), "agent_name": persona.get("name", ""),
        "draw_number": draw_number,
    }
    try:
        fn = CALLERS[provider]
        result = await fn(aio, sem, settings, TASK_SYSTEM_BLOCK, persona_block, user_prompt)
        return {**base, **result, "status": "ok", "error_message": ""}
    except Exception as exc:
        return {**base,
                "forecast_eps": None, "forecast_sales": None,
                "forecast_profit_margin": None, "confidence": None,
                "reasoning": "",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "status": "error", "error_message": str(exc)[:500]}


async def run_pipeline_async(settings, skip_drive_upload=False):
    tz = ZoneInfo(settings.timezone_name)
    run_dt = datetime.now(tz)
    run_id = run_dt.strftime("%Y%m%d_%H%M%S_%Z")
    run_date_pt = run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    date_prefix = run_dt.strftime("%Y%m%d")
    fp = settings.forecast_period or infer_next_quarter_label(run_dt)
    run_dir = settings.local_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "pipeline.log")

    logging.info("=" * 60)
    logging.info("  AGENT-BASED DISTRIBUTION FORECAST PIPELINE")
    logging.info("=" * 60)
    logging.info("Run ID: %s | Period: %s", run_id, fp)

    enabled = []
    if settings.enable_openai and settings.openai_api_key:
        enabled.append(("openai", settings.openai_model, settings.openai_concurrency))
    if settings.enable_anthropic and settings.anthropic_api_key:
        enabled.append(("anthropic", settings.anthropic_model, settings.anthropic_concurrency))
    if settings.enable_gemini and settings.gemini_api_key:
        enabled.append(("gemini", settings.gemini_model, settings.gemini_concurrency))
    if not enabled:
        raise RuntimeError("No providers enabled.")

    personas = load_personas([p[0] for p in enabled])
    if settings.n_draws > 0:
        for prov in list(personas.keys()):
            if len(personas[prov]) > settings.n_draws:
                logging.info("[%s] truncating to first %d personas (have %d)",
                             prov, settings.n_draws, len(personas[prov]))
                personas[prov] = personas[prov][:settings.n_draws]

    payload = fetch_and_save_api_data(settings, run_dt, run_dir)
    if settings.source_api_save_raw:
        save_json(run_dir / "raw_api_payload.json", payload)
    records = normalize_records(payload, settings)
    if not records:
        raise RuntimeError("No usable company records. Check raw_api_payload.json.")
    save_json(run_dir / "normalized_records.json", records)

    current_hash = compute_data_hash(records)
    (run_dir / "data_hash.txt").write_text(current_hash, encoding="utf-8")
    prev_hash, prev_run = find_previous_run_hash(settings.local_output_dir, run_id)
    if prev_hash == current_hash and prev_hash is not None:
        logging.info("=" * 60)
        logging.info("  DATA UNCHANGED since previous run (%s)", prev_run)
        logging.info("  Skipping AI forecasts to save API costs.")
        logging.info("=" * 60)
        summary = {"run_id": run_id, "run_date_pt": run_date_pt, "forecast_period": fp,
                   "status": "skipped_data_unchanged", "previous_run": prev_run,
                   "data_hash": current_hash, "companies": len(records)}
        save_json(run_dir / "run_summary.json", summary)
        if not skip_drive_upload:
            try:
                svc = authorize_drive(settings)
                di = upload_run_folder_to_drive_agents(svc, settings, run_dir, run_dt)
                summary["drive_info"] = di
                save_json(run_dir / "run_summary.json", summary)
                logging.info("Drive upload complete (skip-marker only).")
            except Exception as exc:
                logging.warning("Drive upload failed: %s", exc)
        return run_dir

    providers = [(p[0], p[1], asyncio.Semaphore(p[2])) for p in enabled]
    total = sum(len(records) * len(personas[p[0]]) for p in providers)
    logging.info("-" * 60)
    for pn, pm, _ in providers:
        logging.info("  %-12s -> %s  (%d agents)", pn, pm, len(personas[pn]))
    logging.info("-" * 60)
    logging.info("Total: %d firms × Σ agents = %d tasks", len(records), total)

    tasks = []
    conn = aiohttp.TCPConnector(limit=200, limit_per_host=80)
    sems_by_prov = {p[0]: p[2] for p in providers}
    models_by_prov = {p[0]: p[1] for p in providers}
    async with aiohttp.ClientSession(connector=conn) as aio:
        for rec in records:
            for pn, pm, sem in providers:
                for idx, persona in enumerate(personas[pn], start=1):
                    tasks.append(run_agent_forecast(
                        aio, sem, settings, pn, pm, persona, rec, fp,
                        idx, run_id, run_date_pt,
                    ))
        logging.info("Dispatching %d async tasks...", len(tasks))
        all_rows = list(await asyncio.gather(*tasks))

        # Hard guarantee: keep retrying failed agents until every (firm, agent) succeeds.
        # In-call retries already cover 429/5xx; orchestration retries handle JSON parse
        # failures and exhaustive 429 retries. Cap at MAX_RETRY_ROUNDS to prevent runaway.
        MAX_RETRY_ROUNDS = 10
        for retry_round in range(1, MAX_RETRY_ROUNDS + 1):
            failed_indices = [i for i, r in enumerate(all_rows) if r.get("status") != "ok"]
            if not failed_indices:
                break
            logging.info(
                "Retry round %d/%d: %d failed agents — re-dispatching",
                retry_round, MAX_RETRY_ROUNDS, len(failed_indices),
            )
            retry_tasks = []
            for i in failed_indices:
                old = all_rows[i]
                prov = old["model_provider"]
                persona = next(p for p in personas[prov]
                               if p.get("agent_id") == old["agent_id"])
                record = next(rec for rec in records
                              if rec["company_id"] == old["company_id"])
                retry_tasks.append(run_agent_forecast(
                    aio, sems_by_prov[prov], settings, prov, models_by_prov[prov],
                    persona, record, fp, old["draw_number"], run_id, run_date_pt,
                ))
            retry_results = await asyncio.gather(*retry_tasks)
            for idx, new_r in zip(failed_indices, retry_results):
                all_rows[idx] = new_r

        still_failed = [r for r in all_rows if r.get("status") != "ok"]
        if still_failed:
            logging.error(
                "After %d retry rounds, %d agents still failing: %s",
                MAX_RETRY_ROUNDS, len(still_failed),
                [(r["model_provider"], r["agent_id"], r["error_message"][:80])
                 for r in still_failed[:5]],
            )
            raise RuntimeError(
                f"Failed to reach 300/300 successes after {MAX_RETRY_ROUNDS} "
                f"orchestration retry rounds; {len(still_failed)} agents still failing."
            )
    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    err = len(all_rows) - ok
    cache_hits = sum(r.get("cache_read_tokens", 0) for r in all_rows)
    logging.info("Results: %d ok, %d errors | cache_read_tokens total: %d",
                 ok, err, cache_hits)

    save_csv(run_dir / f"{date_prefix}_per_agent_results.csv", all_rows, DRAW_OUTPUT_FIELDS)
    save_json(run_dir / f"{date_prefix}_per_agent_results.json", all_rows)
    failures = [r for r in all_rows if r.get("status") != "ok"]
    if failures:
        save_json(run_dir / f"{date_prefix}_failures.json", failures)

    dist = compute_distribution(all_rows, run_id, fp)
    save_csv(run_dir / f"{date_prefix}_distribution_summary.csv", dist, DIST_SUMMARY_FIELDS)
    save_json(run_dir / f"{date_prefix}_distribution_summary.json", dist)
    logging.info("Distribution rows: %d", len(dist))

    summary = {
        "run_id": run_id, "run_date_pt": run_date_pt, "forecast_period": fp,
        "n_agents_per_provider": {p[0]: len(personas[p[0]]) for p in providers},
        "providers": [{"name": p[0], "model": p[1]} for p in providers],
        "companies": len(records), "total_tasks": total, "ok": ok, "errors": err,
        "cache_read_tokens_total": cache_hits,
        "data_hash": current_hash,
    }
    save_json(run_dir / "run_summary.json", summary)

    if not skip_drive_upload:
        logging.info("Uploading to Google Drive...")
        svc = authorize_drive(settings)
        di = upload_run_folder_to_drive_agents(svc, settings, run_dir, run_dt)
        summary["drive_info"] = di
        save_json(run_dir / "run_summary.json", summary)
        logging.info("Drive upload complete.")
    else:
        logging.info("Skipping Drive upload.")

    logging.info("=" * 60)
    logging.info("  DONE.  Output: %s", run_dir)
    logging.info("=" * 60)
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Agent-based distribution forecast pipeline")
    p.add_argument("--skip-drive-upload", action="store_true")
    p.add_argument("--n-draws", type=int, default=None,
                   help="Limit to first N agents per provider (default: use all)")
    return p.parse_args()


def _find_up(name, start):
    """Walk up from `start` until a file named `name` is found. Returns the Path or None."""
    for d in [start, *start.parents]:
        p = d / name
        if p.exists():
            return p
    return None


def main():
    try:
        settings = load_settings()
        # Keep new pipeline's output separate from the original pipeline
        settings.local_output_dir = (SCRIPT_DIR / "output").resolve()
        # Share Drive credentials with the original pipeline. Walk up from SCRIPT_DIR so
        # this works both inside a git worktree (creds live at the real project root) and
        # after merging back to main (creds live at the worktree's PARENT_DIR).
        cred_path = _find_up("credentials.json", SCRIPT_DIR) or (PARENT_DIR / "credentials.json")
        tok_path = _find_up("token.json", SCRIPT_DIR) or (PARENT_DIR / "token.json")
        settings.google_drive_credentials_file = cred_path.resolve()
        settings.google_drive_token_file = tok_path.resolve()
        # Bump max_tokens floor — reasoning adds ~200-300 output tokens; the original
        # pipeline's 400-token cap risks mid-string JSON truncation in the agent version.
        settings.openai_max_completion_tokens = max(settings.openai_max_completion_tokens, 800)
        settings.anthropic_max_tokens = max(settings.anthropic_max_tokens, 800)
        settings.gemini_max_output_tokens = max(settings.gemini_max_output_tokens, 800)
        args = parse_args()
        if args.n_draws is not None:
            settings.n_draws = args.n_draws
        run_dir = asyncio.run(run_pipeline_async(
            settings, skip_drive_upload=args.skip_drive_upload,
        ))
        print(f"DONE: {run_dir}")
        return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
