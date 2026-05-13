#!/usr/bin/env python3
"""
Agent Persona Training (One-time)
==================================
Author: Rui Sun

Generates N fixed analyst personas per LLM provider using a small sample of real
companies from the data API as context, then freezes the personas to JSON.

Run ONCE. The output files (agents/{provider}_agents.json) are loaded by every
nightly forecast run — never regenerated unless you explicitly want a new panel.

Usage:
  python train_agents.py                       # default: 100 personas per provider
  python train_agents.py --n-personas 50
  python train_agents.py --providers anthropic gemini
"""
from __future__ import annotations
__author__ = "Rui Sun"
import argparse, asyncio, json, logging, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

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
    load_settings, fetch_source_payload_sync, normalize_records,
    parse_json_from_text, setup_logging,
)

AGENTS_DIR = SCRIPT_DIR / "agents"
BATCH_SIZE = 10
PERSONA_REQUIRED_FIELDS = [
    "name", "years_experience", "sector_focus", "methodology",
    "optimism_bias", "guidance_weight", "risk_tolerance", "notable_beliefs",
]

META_SYSTEM = (
    "You are designing a heterogeneous panel of sell-side equity research analysts "
    "for a Monte Carlo earnings forecast study.\n\n"
    "Generate distinct, plausible human analyst personas that will be used independently "
    "to produce next-quarter forecasts of EPS, sales, and profit margin for a set of public "
    "companies.\n\n"
    "Diversity matters. The panel as a whole should span the realistic spectrum of:\n"
    "- experience (junior 3y to veteran 25y)\n"
    "- sector specialty (technology, healthcare, consumer_discretionary, consumer_staples, "
    "energy, financials, industrials, materials, real_estate, utilities, communications, diversified)\n"
    "- methodology (DCF-heavy, multiples-heavy, momentum/technical, top-down macro, "
    "bottom-up fundamental, contrarian/value, quant/factor-based)\n"
    "- optimism/pessimism bias (roughly normal-distributed around 'balanced')\n"
    "- trust in management guidance (skeptical -> high trust)\n"
    "- risk tolerance (conservative -> aggressive)\n\n"
    "Avoid clones. Avoid one-dimensional caricatures. Each persona should feel like "
    "a real, opinionated human analyst with a coherent forecasting style."
)

PERSONA_SCHEMA = """Each persona object MUST have exactly these fields:
- name: short descriptive string (e.g. "Mid-Cycle Tech Generalist"). Under 60 chars. Each must be unique.
- years_experience: integer 3-25
- sector_focus: list of 1-3 sector strings
- methodology: 1 sentence describing forecasting approach
- optimism_bias: one of "strongly_pessimistic" | "pessimistic" | "balanced" | "optimistic" | "strongly_optimistic"
- guidance_weight: one of "skeptical" | "neutral" | "high_trust"
- risk_tolerance: one of "conservative" | "moderate" | "aggressive"
- notable_beliefs: 1-2 sentence description of distinctive views (under 250 chars)
- signature_phrases: list of 2-3 short phrases this analyst tends to use
"""


def build_sample_context(records, n_companies=4, n_history=3):
    samples = []
    for rec in records[:n_companies]:
        history_short = rec.get("history", [])[-n_history:]
        samples.append({
            "ticker": rec.get("ticker"),
            "company_name": rec.get("company_name"),
            "currency": rec.get("currency"),
            "recent_history": history_short,
            "context_snippet": (rec.get("context", "") or "")[:200],
        })
    return json.dumps(samples, indent=2, default=str)


def build_user_prompt(n_personas, batch_idx, total_batches, sample_context):
    return (
        f"Generate exactly {n_personas} analyst personas (batch {batch_idx} of {total_batches}).\n\n"
        f"{PERSONA_SCHEMA}\n"
        f"Context — sample of companies the analysts will forecast:\n{sample_context}\n\n"
        f"Return ONLY a JSON object with key \"personas\" containing the array of "
        f"{n_personas} persona objects. No markdown, no preamble, no trailing prose."
    )


async def gen_openai_batch(aio, settings, sys_p, usr_p):
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        "max_completion_tokens": 4000,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}",
               "Content-Type": "application/json"}
    for attempt in range(5):
        async with aio.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(min(2 ** attempt * 3, 60))
                continue
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI {resp.status}: {json.dumps(body)[:600]}")
            text = body["choices"][0]["message"]["content"]
            return parse_json_from_text(text)
    raise RuntimeError("OpenAI: max retries")


async def gen_anthropic_batch(aio, settings, sys_p, usr_p):
    payload = {
        "model": settings.anthropic_model,
        "max_tokens": 4000,
        "system": sys_p,
        "messages": [{"role": "user", "content": usr_p}],
    }
    headers = {"x-api-key": settings.anthropic_api_key,
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    for attempt in range(5):
        async with aio.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(min(2 ** attempt * 3, 60))
                continue
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"Anthropic {resp.status}: {json.dumps(body)[:600]}")
            text = "".join(b.get("text", "") for b in body.get("content", [])
                           if b.get("type") == "text")
            return parse_json_from_text(text)
    raise RuntimeError("Anthropic: max retries")


async def gen_gemini_batch(aio, settings, sys_p, usr_p):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}")
    payload = {
        "system_instruction": {"parts": [{"text": sys_p}]},
        "contents": [{"parts": [{"text": usr_p}]}],
        "generationConfig": {
            "maxOutputTokens": 4000,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(5):
        async with aio.post(url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 429:
                await asyncio.sleep(min(2 ** attempt * 3, 60))
                continue
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"Gemini {resp.status}: {json.dumps(body)[:600]}")
            text = "".join(p.get("text", "") for c in body.get("candidates", [])
                           for p in c.get("content", {}).get("parts", []))
            return parse_json_from_text(text)
    raise RuntimeError("Gemini: max retries")


GENERATORS = {
    "openai": gen_openai_batch,
    "anthropic": gen_anthropic_batch,
    "gemini": gen_gemini_batch,
}


def validate_personas(raw, provider):
    if isinstance(raw, dict) and "personas" in raw:
        raw = raw["personas"]
    if not isinstance(raw, list):
        logging.warning("[%s] response not a list (got %s); skipping batch",
                        provider, type(raw).__name__)
        return []
    out = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        missing = [f for f in PERSONA_REQUIRED_FIELDS if f not in p]
        if missing:
            logging.warning("[%s] persona missing %s — skipping (name=%s)",
                            provider, missing, p.get("name", "?"))
            continue
        sf = p.get("sector_focus", [])
        if isinstance(sf, str):
            sf = [sf]
        p["sector_focus"] = [str(s) for s in (sf or [])]
        sp = p.get("signature_phrases", [])
        if isinstance(sp, str):
            sp = [sp]
        p["signature_phrases"] = [str(s) for s in (sp or [])]
        out.append(p)
    return out


def dedupe_personas(personas):
    seen = set()
    out = []
    for p in personas:
        key = p.get("name", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


async def generate_provider_personas(aio, provider, settings, n_target, sample_context):
    """Keep dispatching batches until n_target unique personas are collected.

    Hard guarantee: returns exactly n_target personas (or raises if it stalls — i.e. the
    LLM keeps producing duplicates with zero new additions for 2 consecutive rounds).
    Soft cap at 20 rounds purely as a runaway-prevention backstop; in practice 1-2 rounds
    is enough at n_target=100.
    """
    fn = GENERATORS[provider]
    collected = []
    round_idx = 0
    stalled_rounds = 0
    MAX_ROUNDS = 20
    while len(collected) < n_target and round_idx < MAX_ROUNDS:
        round_idx += 1
        deficit = n_target - len(collected)
        batch_n = min(BATCH_SIZE, deficit)
        n_batches = max(1, (deficit + batch_n - 1) // batch_n)
        logging.info("[%s] round %d — dispatching %d batches x %d personas (deficit %d)",
                     provider, round_idx, n_batches, batch_n, deficit)
        tasks = []
        for b in range(1, n_batches + 1):
            global_idx = b + (round_idx - 1) * 100
            usr_p = build_user_prompt(batch_n, global_idx, 100, sample_context)
            tasks.append(fn(aio, settings, META_SYSTEM, usr_p))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        before = len(collected)
        for r in results:
            if isinstance(r, Exception):
                logging.warning("[%s] batch failed: %s", provider, str(r)[:200])
                continue
            collected.extend(validate_personas(r, provider))
        collected = dedupe_personas(collected)
        gained = len(collected) - before
        logging.info("[%s] %d unique personas after round %d (+%d this round, target %d)",
                     provider, len(collected), round_idx, gained, n_target)
        if gained == 0:
            stalled_rounds += 1
            if stalled_rounds >= 2:
                raise RuntimeError(
                    f"[{provider}] stalled at {len(collected)}/{n_target} personas "
                    f"after 2 consecutive zero-progress rounds — LLM may be returning "
                    f"only duplicates."
                )
        else:
            stalled_rounds = 0
    if len(collected) < n_target:
        raise RuntimeError(
            f"[{provider}] hit {MAX_ROUNDS}-round soft cap with only "
            f"{len(collected)}/{n_target} personas."
        )
    final = collected[:n_target]
    for i, p in enumerate(final, start=1):
        p["agent_id"] = f"{provider}_{i:03d}"
    return final


async def main_async(args):
    settings = load_settings()
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(AGENTS_DIR / "train_agents.log")

    tz = ZoneInfo(settings.timezone_name)
    run_dt = datetime.now(tz)

    logging.info("=" * 60)
    logging.info("  AGENT PERSONA TRAINING (one-time)")
    logging.info("=" * 60)
    logging.info("Target: %d personas per provider", args.n_personas)
    logging.info("Requested providers: %s", args.providers)
    logging.info("Models: openai=%s anthropic=%s gemini=%s",
                 settings.openai_model, settings.anthropic_model, settings.gemini_model)

    logging.info("Fetching sample data from API for context...")
    payload = fetch_source_payload_sync(settings, run_dt)
    records = normalize_records(payload, settings)
    if not records:
        raise RuntimeError("No usable records from API")
    sample_context = build_sample_context(records, n_companies=4, n_history=3)
    logging.info("Sample context: %d companies, %d chars",
                 min(4, len(records)), len(sample_context))

    available = {
        "openai": settings.enable_openai and bool(settings.openai_api_key),
        "anthropic": settings.enable_anthropic and bool(settings.anthropic_api_key),
        "gemini": settings.enable_gemini and bool(settings.gemini_api_key),
    }
    to_run = [p for p in args.providers if available.get(p)]
    skipped = [p for p in args.providers if not available.get(p)]
    if skipped:
        logging.warning("Skipping (disabled or missing key): %s", skipped)
    if not to_run:
        raise RuntimeError("No providers available. Check .env and --providers.")

    conn = aiohttp.TCPConnector(limit=60, limit_per_host=20)
    async with aiohttp.ClientSession(connector=conn) as aio:
        coros = {p: generate_provider_personas(aio, p, settings, args.n_personas, sample_context)
                 for p in to_run}
        results = await asyncio.gather(*coros.values(), return_exceptions=True)

    summary = {}
    for provider, result in zip(coros.keys(), results):
        if isinstance(result, Exception):
            logging.error("[%s] FAILED: %s", provider, result)
            summary[provider] = {"status": "failed", "error": str(result)[:300]}
            continue
        out_path = AGENTS_DIR / f"{provider}_agents.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("[%s] saved %d personas -> %s", provider, len(result), out_path.name)
        summary[provider] = {"status": "ok", "n_personas": len(result), "file": out_path.name}

    (AGENTS_DIR / "train_summary.json").write_text(
        json.dumps({
            "trained_at": run_dt.isoformat(),
            "n_target_per_provider": args.n_personas,
            "models_used": {
                "openai": settings.openai_model,
                "anthropic": settings.anthropic_model,
                "gemini": settings.gemini_model,
            },
            "results": summary,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    logging.info("=" * 60)
    logging.info("  DONE. Agents directory: %s", AGENTS_DIR)
    for p, s in summary.items():
        logging.info("    %-12s %s", p, s)
    logging.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Generate fixed analyst personas (one-time)")
    p.add_argument("--n-personas", type=int, default=100,
                   help="Personas per provider (default 100)")
    p.add_argument("--providers", nargs="+",
                   default=["openai", "anthropic", "gemini"],
                   choices=["openai", "anthropic", "gemini"])
    return p.parse_args()


def main():
    try:
        args = parse_args()
        asyncio.run(main_async(args))
        return 0
    except Exception as exc:
        logging.exception("train_agents failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
