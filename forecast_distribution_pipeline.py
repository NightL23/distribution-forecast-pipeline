#!/usr/bin/env python3
"""
Multi-Model Distribution Forecast Pipeline  (Final Version)
============================================================
Author: Rui Sun

For every firm from the professor's API, fire N independent forecast prompts
(default 100) to each of three LLM providers simultaneously.
Forecasts: EPS, Sales, Profit Margin
Distribution stats: mean, median, std, skewness, kurtosis, min, max, percentiles

Behavior:
  - On first run (no previous data hash on disk): executes the full forecast pipeline.
  - On subsequent runs: if the incoming data from the API is byte-for-byte identical
    to the previous successful run, skips all AI calls to save cost. If the data
    has changed at all, or no previous hash is found, runs the full forecast.
  - All output CSVs are prefixed with the date (YYYYMMDD).
"""
from __future__ import annotations
__author__ = "Rui Sun"
import argparse, asyncio, csv, io, json, logging, math, mimetypes, os, re
import statistics, sys, time, zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo
import aiohttp, requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build as google_build
from googleapiclient.http import MediaFileUpload
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openpyxl import load_workbook

TZ_PT = ZoneInfo("America/Los_Angeles")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DRAW_OUTPUT_FIELDS = [
    "run_id","run_date_pt","forecast_period","company_id","ticker","company_name","currency",
    "model_provider","model_name","draw_number",
    "forecast_eps","forecast_sales","forecast_profit_margin","confidence",
    "status","error_message","prompt_tokens","completion_tokens","total_tokens",
]
DIST_SUMMARY_FIELDS = [
    "run_id","forecast_period","company_id","ticker","company_name","currency",
    "model_provider","model_name","n_draws_ok",
    "eps_mean","eps_median","eps_std","eps_skewness","eps_kurtosis",
    "eps_min","eps_max","eps_p5","eps_p10","eps_p25","eps_p75","eps_p90","eps_p95",
    "sales_mean","sales_median","sales_std","sales_skewness","sales_kurtosis",
    "sales_min","sales_max","sales_p5","sales_p10","sales_p25","sales_p75","sales_p90","sales_p95",
    "pm_mean","pm_median","pm_std","pm_skewness","pm_kurtosis",
    "pm_min","pm_max","pm_p5","pm_p10","pm_p25","pm_p75","pm_p90","pm_p95",
    "confidence_mean",
]

@dataclass
class Settings:
    n_draws: int
    openai_api_key: str; openai_model: str; openai_max_completion_tokens: int
    openai_temperature: float; openai_concurrency: int
    anthropic_api_key: str; anthropic_model: str; anthropic_max_tokens: int
    anthropic_temperature: float; anthropic_concurrency: int
    gemini_api_key: str; gemini_model: str; gemini_max_output_tokens: int
    gemini_temperature: float; gemini_concurrency: int
    data_api_url: str; data_api_method: str; data_api_timeout_seconds: int
    data_api_headers: Dict[str,Any]; data_api_params: Dict[str,Any]
    data_api_body: Dict[str,Any]; data_api_records_path: str
    forecast_period: str; max_companies_per_run: int
    prompt_max_history_points: int; prompt_max_note_chars: int
    prompt_max_company_context_chars: int; prompt_include_eps_history: bool
    local_output_dir: Path
    google_drive_credentials_file: Path; google_drive_token_file: Path
    google_drive_root_folder_id: str; google_drive_root_folder_name: str
    timezone_name: str; source_api_save_raw: bool
    enable_openai: bool; enable_anthropic: bool; enable_gemini: bool

def env_bool(name, default):
    v = os.getenv(name)
    return v.strip().lower() in {"1","true","yes","y","on"} if v is not None else default

def env_json(name, default=None):
    raw = os.getenv(name, "")
    if not raw: return default or {}
    v = json.loads(raw)
    if not isinstance(v, dict): raise ValueError(f"{name} must be JSON object.")
    return v

def load_settings():
    load_dotenv()
    url = os.getenv("DATA_API_URL","").strip()
    if not url: raise ValueError("DATA_API_URL required.")
    h = env_json("DATA_API_HEADERS_JSON",{})
    p = env_json("DATA_API_PARAMS_JSON",{})
    b = env_json("DATA_API_BODY_JSON",{})
    bearer = os.getenv("DATA_API_BEARER_TOKEN","").strip()
    if bearer and "Authorization" not in h: h["Authorization"]=f"Bearer {bearer}"
    akn = os.getenv("DATA_API_KEY_HEADER_NAME","").strip()
    akv = os.getenv("DATA_API_KEY","").strip()
    if akn and akv and akn not in h: h[akn]=akv
    return Settings(
        n_draws=int(os.getenv("N_DRAWS","100")),
        openai_api_key=os.getenv("OPENAI_API_KEY","").strip(),
        openai_model=os.getenv("OPENAI_MODEL","gpt-4o-mini").strip(),
        openai_max_completion_tokens=int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS","400")),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE","0.7")),
        openai_concurrency=int(os.getenv("OPENAI_CONCURRENCY","50")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY","").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL","claude-haiku-4-5-20251001").strip(),
        anthropic_max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS","400")),
        anthropic_temperature=float(os.getenv("ANTHROPIC_TEMPERATURE","0.7")),
        anthropic_concurrency=int(os.getenv("ANTHROPIC_CONCURRENCY","40")),
        gemini_api_key=os.getenv("GEMINI_API_KEY","").strip(),
        gemini_model=os.getenv("GEMINI_MODEL","gemini-2.5-flash-lite").strip(),
        gemini_max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS","400")),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE","0.7")),
        gemini_concurrency=int(os.getenv("GEMINI_CONCURRENCY","50")),
        data_api_url=url, data_api_method=os.getenv("DATA_API_METHOD","GET").strip().upper(),
        data_api_timeout_seconds=int(os.getenv("DATA_API_TIMEOUT_SECONDS","120")),
        data_api_headers=h, data_api_params=p, data_api_body=b,
        data_api_records_path=os.getenv("DATA_API_RECORDS_PATH","").strip(),
        forecast_period=os.getenv("FORECAST_PERIOD","").strip(),
        max_companies_per_run=int(os.getenv("MAX_COMPANIES_PER_RUN","0")),
        prompt_max_history_points=int(os.getenv("PROMPT_MAX_HISTORY_POINTS","8")),
        prompt_max_note_chars=int(os.getenv("PROMPT_MAX_NOTE_CHARS","160")),
        prompt_max_company_context_chars=int(os.getenv("PROMPT_MAX_COMPANY_CONTEXT_CHARS","600")),
        prompt_include_eps_history=env_bool("PROMPT_INCLUDE_EPS_HISTORY",False),
        local_output_dir=Path(os.getenv("LOCAL_OUTPUT_DIR","./output")).resolve(),
        google_drive_credentials_file=Path(os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE","./credentials.json")).resolve(),
        google_drive_token_file=Path(os.getenv("GOOGLE_DRIVE_TOKEN_FILE","./token.json")).resolve(),
        google_drive_root_folder_id=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID","").strip(),
        google_drive_root_folder_name=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_NAME","nightly_forecasts").strip() or "nightly_forecasts",
        timezone_name=os.getenv("TIMEZONE_NAME","America/Los_Angeles").strip(),
        source_api_save_raw=env_bool("SOURCE_API_SAVE_RAW",True),
        enable_openai=env_bool("ENABLE_OPENAI",True),
        enable_anthropic=env_bool("ENABLE_ANTHROPIC",True),
        enable_gemini=env_bool("ENABLE_GEMINI",True),
    )

# ── Utilities ──
def infer_next_quarter_label(ts):
    q=((ts.month-1)//3)+1; nq=q+1; y=ts.year
    if nq==5: nq=1; y+=1
    return f"{y}Q{nq}"

def setup_logging(log_file):
    log_file.parent.mkdir(parents=True,exist_ok=True)
    root=logging.getLogger(); root.handlers.clear(); root.setLevel(logging.INFO)
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh=logging.FileHandler(log_file,encoding="utf-8"); fh.setFormatter(fmt); root.addHandler(fh)

def render_template_in_obj(value, context):
    if isinstance(value,dict): return {k:render_template_in_obj(v,context) for k,v in value.items()}
    if isinstance(value,list): return [render_template_in_obj(v,context) for v in value]
    if isinstance(value,str):
        out=value
        for key,rep in context.items(): out=out.replace(f"{{{{{key}}}}}",rep)
        return out
    return value

def deep_get(obj, path, default=None):
    if not path: return default
    cur=obj
    for p in path.split("."):
        if isinstance(cur,dict) and p in cur: cur=cur[p]
        else: return default
    return cur

def maybe_number(value):
    if value is None or value=="": return None
    if isinstance(value,(int,float)): return float(value)
    if isinstance(value,str):
        raw=value.strip().replace(",","")
        if raw in {"","NA","N/A","null","None","nan"}: return None
        try: return float(raw)
        except ValueError: return None
    return None

def first_nonempty(d, keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None,"",[],{}): return d[k]
    return default

def first_list(d, keys):
    for k in keys:
        v=d.get(k)
        if isinstance(v,list): return v
    return None

def sanitize_text(value, max_chars):
    if value is None: return ""
    return re.sub(r"\s+"," ",str(value)).strip()[:max_chars]

def period_sort_key(period):
    if not period: return (0,0,0,"")
    p=period.strip()
    m=re.match(r"^(\d{4})Q([1-4])$",p,re.IGNORECASE)
    if m: return (int(m.group(1)),int(m.group(2)),0,p)
    for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y-%m","%Y/%m"):
        try: dt=datetime.strptime(p,fmt); return (dt.year,dt.month,dt.day,p)
        except ValueError: pass
    return (0,0,0,p)

def extract_company_level_context(record, max_chars):
    pieces=[]
    for key in ["summary","context","guidance","management_commentary","notes","comment","description","sector","multi_class"]:
        if key in record and record[key] not in (None,""): pieces.append(f"{key}: {sanitize_text(record[key],max_chars)}")
    return " | ".join(pieces)[:max_chars]

def extract_non_eps_financials(entry, max_items=6):
    excluded={"company_id","id","gvkey","entity_id","issuer_id","ticker","symbol","tic","company_ticker",
        "company_name","name","company","issuer","full_name","currency","ccy",
        "period","fiscal_period","quarter","fiscalquarter","fiscal_quarter",
        "period_end","periodend","date","as_of_date","fiscal_date",
        "fiscal_year","fiscalyear","year","quarter_num",
        "note","notes","summary","comment","guidance","management_commentary","description",
        "sector","multi_class","status",
        "sales","saleq","revenue","revenues","net_sales","total_revenue"}
    out={}
    for key,value in entry.items():
        kn=re.sub(r"[^a-z0-9]+","_",str(key).lower()).strip("_")
        if not kn or kn in excluded or "eps" in kn: continue
        num=maybe_number(value)
        if num is None: continue
        out[kn]=num
        if len(out)>=max_items: break
    return out

# ── Normalization ──
def normalize_history_entry(entry):
    if not isinstance(entry,dict): return None
    period=first_nonempty(entry,["period","fiscal_period","quarter","fiscalQuarter","fiscal_quarter","period_end","periodEnd","date","as_of_date","fiscal_date"],"")
    fy=first_nonempty(entry,["fiscal_year","fiscalYear","year"],"")
    fq=first_nonempty(entry,["fiscal_quarter","fiscalQuarter","quarter_num"],"")
    if not period and fy and fq:
        period=f"{str(fy).strip().replace('.0','')}Q{str(fq).strip().replace('.0','')}"
    eps=maybe_number(first_nonempty(entry,["eps","adj_eps","adjusted_eps","eps_diluted","diluted_eps","reported_eps"],None))
    sales=maybe_number(first_nonempty(entry,["sales","saleq","revenue","revenues","net_sales","total_revenue"],None))
    if eps is None and sales is None: return None
    note=sanitize_text(first_nonempty(entry,["note","notes","summary","comment","guidance"],""),160)
    extras=extract_non_eps_financials(entry)
    return {"period":str(period).strip(),"eps":eps,"sales":sales,"extras":extras,"note":note}

def normalize_company_record(record):
    if not isinstance(record,dict): return None
    cid=str(first_nonempty(record,["company_id","id","gvkey","entity_id","issuer_id"],"")).strip()
    ticker=str(first_nonempty(record,["ticker","symbol","tic","company_ticker"],"")).strip()
    cname=str(first_nonempty(record,["company_name","name","company","issuer","full_name"],"")).strip()
    currency=str(first_nonempty(record,["currency","ccy"],"")).strip()
    hraw=first_list(record,["history","financials","quarters","periods","observations","series"])
    history=[]
    if hraw:
        for item in hraw:
            n=normalize_history_entry(item)
            if n: history.append(n)
    else:
        s=normalize_history_entry(record)
        if s: history=[s]
    if not history: return None
    history=sorted(history,key=lambda x:period_sort_key(x.get("period","")))
    if not ticker and cname: ticker=re.sub(r"\s+","_",cname.upper())[:32]
    if not cid: cid=ticker or cname or f"company_{abs(hash(json.dumps(record,sort_keys=True)))%10_000_000}"
    return {"company_id":cid,"ticker":ticker,"company_name":cname or ticker or cid,
            "currency":currency,"history":history,
            "context":extract_company_level_context(record,600),
            "raw_meta":{"source_keys":sorted(list(record.keys()))[:50]}}

def normalize_flat_records(records):
    groups={}; grouped_history=defaultdict(list)
    for row in records:
        if not isinstance(row,dict): continue
        cid=str(first_nonempty(row,["company_id","id","gvkey","entity_id","issuer_id"],"")).strip()
        ticker=str(first_nonempty(row,["ticker","symbol","tic","company_ticker"],"")).strip()
        cname=str(first_nonempty(row,["company_name","name","company","issuer","full_name"],"")).strip()
        key=cid or ticker or cname
        if not key: continue
        hist=normalize_history_entry(row)
        if hist is None: continue
        grouped_history[key].append(hist)
        if key not in groups:
            groups[key]={"company_id":cid or key,"ticker":ticker,"company_name":cname or ticker or key,
                "currency":str(first_nonempty(row,["currency","ccy"],"")).strip(),
                "context":extract_company_level_context(row,600),
                "raw_meta":{"source_keys":sorted(list(row.keys()))[:50]}}
    out=[]
    for key,meta in groups.items():
        history=sorted(grouped_history[key],key=lambda x:period_sort_key(x.get("period","")))
        out.append({**meta,"history":history})
    return out

def infer_records_container(payload):
    if isinstance(payload,list): return payload
    if not isinstance(payload,dict): raise ValueError("Payload must be list or dict.")
    for key in ["records","data","companies","items","results","observations"]:
        v=payload.get(key)
        if isinstance(v,list): return v
        if isinstance(v,dict):
            for sk in ["records","items","companies","results"]:
                sv=v.get(sk)
                if isinstance(sv,list): return sv
    for _,v in payload.items():
        if isinstance(v,list): return v
    raise ValueError("Could not find records in payload.")

def normalize_records(payload, settings):
    records=deep_get(payload,settings.data_api_records_path,None) if settings.data_api_records_path else None
    if records is None: records=infer_records_container(payload)
    if not isinstance(records,list): raise ValueError("Records path not a list.")
    logging.info("Raw records from API: %d",len(records))
    if records and isinstance(records[0],dict):
        logging.info("Sample keys: %s",list(records[0].keys())[:12])
    has_nested=any(isinstance(r,dict) and first_list(r,["history","financials","quarters","periods","observations","series"]) for r in records[:50])
    if has_nested:
        logging.info("Nested history format detected.")
        normalized=[n for r in records if (n:=normalize_company_record(r)) is not None]
    else:
        logging.info("Flat format detected. Grouping by company...")
        normalized=normalize_flat_records(records)
    normalized=[r for r in normalized if r.get("history")]
    normalized.sort(key=lambda x:(x.get("ticker",""),x.get("company_name",""),x.get("company_id","")))
    if settings.max_companies_per_run>0: normalized=normalized[:settings.max_companies_per_run]
    logging.info("Companies with history: %d",len(normalized))
    if normalized:
        s=normalized[0]
        logging.info("Sample: ticker=%s, name=%s, history_pts=%d",s.get("ticker"),s.get("company_name"),len(s.get("history",[])))
    return normalized

# ── File parsing ──
def parse_csv_bytes(data, delimiter=","):
    text=data.decode("utf-8-sig",errors="replace")
    reader=csv.DictReader(io.StringIO(text),delimiter=delimiter)
    rows=[dict(row) for row in reader]
    logging.info("CSV parsed: %d rows",len(rows))
    return rows

def parse_xlsx_bytes(data, sheet_name=""):
    wb=load_workbook(filename=io.BytesIO(data),read_only=True,data_only=True)
    names=wb.sheetnames
    norm_=lambda s:re.sub(r"[^a-z0-9]+","",s.lower())
    if sheet_name:
        pref=norm_(sheet_name)
        selected=next((n for n in names if norm_(n)==pref or pref in norm_(n)),names[0])
    else:
        selected=sorted(names,key=lambda n:(0 if "standardized" in norm_(n) else 1,1 if "benchmark" in norm_(n) else 0,norm_(n)))[0]
    logging.info("Excel sheet: %s",selected)
    ws=wb[selected]; rows=list(ws.iter_rows(values_only=True))
    if not rows: return []
    headers=["" if v is None else str(v).strip() for v in rows[0]]
    return [{h:row[i] if i<len(row) else None for i,h in enumerate(headers) if h} for row in rows[1:] if any(v not in (None,"") for v in row)]

def parse_payload_from_bytes(data, filename="", content_type=""):
    ln=filename.lower(); lc=content_type.lower()
    sn=os.getenv("DATA_API_SHEET_NAME","").strip()
    if ln.endswith(".json") or "application/json" in lc: return json.loads(data.decode("utf-8-sig",errors="replace"))
    if ln.endswith(".csv") or "text/csv" in lc: return parse_csv_bytes(data)
    if ln.endswith(".tsv") or "tab-separated" in lc: return parse_csv_bytes(data,delimiter="\t")
    if ln.endswith((".xlsx",".xlsm",".xls")) or "spreadsheetml" in lc or "excel" in lc: return parse_xlsx_bytes(data,sheet_name=sn)
    try: return json.loads(data.decode("utf-8-sig",errors="replace"))
    except Exception: pass
    try: return parse_csv_bytes(data)
    except Exception: pass
    raise RuntimeError(f"Unsupported format: {filename!r}")

def fetch_source_payload_sync(settings, run_dt_pt):
    retry=Retry(total=5,backoff_factor=1.2,status_forcelist=[429,500,502,503,504],allowed_methods=["GET","POST"],raise_on_status=False)
    adapter=HTTPAdapter(max_retries=retry); session=requests.Session()
    session.mount("https://",adapter); session.mount("http://",adapter)
    ctx={"run_date":run_dt_pt.strftime("%Y-%m-%d"),"run_datetime_iso":run_dt_pt.isoformat(),
         "forecast_period":settings.forecast_period or infer_next_quarter_label(run_dt_pt)}
    hdrs=render_template_in_obj(settings.data_api_headers,ctx)
    prms=render_template_in_obj(settings.data_api_params,ctx)
    bdy=render_template_in_obj(settings.data_api_body,ctx)
    logging.info("Fetching from API: %s",settings.data_api_url)
    if settings.data_api_method=="GET":
        resp=session.get(settings.data_api_url,headers=hdrs,params=prms,timeout=settings.data_api_timeout_seconds)
    elif settings.data_api_method=="POST":
        resp=session.post(settings.data_api_url,headers=hdrs,params=prms,json=bdy,timeout=settings.data_api_timeout_seconds)
    else: raise ValueError("Method must be GET or POST.")
    if resp.status_code>=400: raise RuntimeError(f"API failed: {resp.status_code}: {resp.text[:500]}")
    ct=resp.headers.get("Content-Type","").lower(); cd=resp.headers.get("Content-Disposition","").lower()
    if "application/zip" in ct or "x-zip" in ct or ".zip" in cd or resp.content[:2]==b"PK":
        preferred=os.getenv("DATA_API_ZIP_MEMBER","").strip()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members=[n for n in zf.namelist() if not n.endswith("/")]
            logging.info("ZIP members: %s",members)
            if preferred:
                member=None
                for f in members:
                    fn=Path(f).stem.lower(); ff=Path(f).name.lower(); pl=preferred.lower()
                    if ff==pl or fn==pl or pl in fn: member=f; break
                if not member: member=members[0]; logging.warning("ZIP member %s not found, using %s",preferred,member)
            else:
                member=sorted(members,key=lambda n:(0 if "standardized" in n.lower() else 1,1 if "benchmark" in n.lower() else 0,n.lower()))[0]
            logging.info("Using ZIP member: %s",member)
            return parse_payload_from_bytes(zf.read(member),filename=member)
    try: return resp.json()
    except json.JSONDecodeError: return parse_payload_from_bytes(resp.content,filename=settings.data_api_url,content_type=ct)

# ── Prompt building ──
def _format_extras(extras,max_chars=180):
    if not extras: return ""
    return sanitize_text(", ".join(f"{k}={v}" for k,v in extras.items()),max_chars)

def build_history_block(record,max_points,note_chars,include_eps=False):
    history=record.get("history",[])[-max_points:]
    lines=["period | eps | sales | extra_financials | note"] if include_eps else ["period | sales | extra_financials | note"]
    for item in history:
        sales="" if item.get("sales") is None else f"{item['sales']}"
        extras=_format_extras(item.get("extras",{}))
        note=sanitize_text(item.get("note",""),note_chars)
        if include_eps:
            eps="" if item.get("eps") is None else f"{item['eps']}"
            lines.append(f"{item.get('period','')} | {eps} | {sales} | {extras} | {note}")
        else:
            lines.append(f"{item.get('period','')} | {sales} | {extras} | {note}")
    return "\n".join(lines),len(history)

def build_prompts(record, forecast_period, settings):
    system_prompt=(
        "You are a careful financial forecasting assistant. "
        "Forecast the next period for exactly one company using only that company's provided history. "
        "Do not use other companies, sector averages, or external knowledge. "
        "Prefer stable extrapolation from recent trend and seasonality when visible. "
        "If data are sparse, stay conservative and lean on the most recent observations. "
        "Return ONLY valid JSON with these exact keys: "
        "company_id (string), ticker (string), company_name (string), forecast_period (string), "
        "forecast_eps (number), forecast_sales (number), "
        "forecast_profit_margin (number as decimal, e.g. 0.15 means 15 percent), "
        "confidence (number between 0 and 1). "
        "No markdown, no code fences, no reasoning, just the raw JSON object."
    )
    hblock,hpts=build_history_block(record,max_points=settings.prompt_max_history_points,
        note_chars=settings.prompt_max_note_chars,include_eps=settings.prompt_include_eps_history)
    ctx=sanitize_text(record.get("context",""),settings.prompt_max_company_context_chars)
    user_prompt=(
        f"Company ID: {record.get('company_id','')}\n"
        f"Ticker: {record.get('ticker','')}\n"
        f"Company name: {record.get('company_name','')}\n"
        f"Currency: {record.get('currency','')}\n"
        f"Forecast period: {forecast_period}\n\n"
        f"Recent financial history:\n{hblock}\n\n"
        f"Additional context: {ctx or 'None'}\n\nReturn the JSON forecast now."
    )
    return system_prompt,user_prompt,hpts

def parse_json_from_text(text):
    text=text.strip()
    if text.startswith("```"):
        lines=text.split("\n")
        lines=[l for l in lines if not l.strip().startswith("```")]
        text="\n".join(lines).strip()
    try: return json.loads(text)
    except json.JSONDecodeError:
        s=text.find("{"); e=text.rfind("}")
        if s!=-1 and e!=-1 and e>s: return json.loads(text[s:e+1])
        raise

# ── Async LLM callers ──
async def call_openai(aio_session,sem,settings,sys_p,usr_p):
    async with sem:
        payload={"model":settings.openai_model,"messages":[{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
            "temperature":settings.openai_temperature,"max_completion_tokens":settings.openai_max_completion_tokens}
        headers={"Authorization":f"Bearer {settings.openai_api_key}","Content-Type":"application/json"}
        for attempt in range(5):
            async with aio_session.post("https://api.openai.com/v1/chat/completions",json=payload,headers=headers,timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status==429: await asyncio.sleep(min(2**attempt*2,60)); continue
                body=await resp.json()
                if resp.status>=400: raise RuntimeError(f"OpenAI {resp.status}: {json.dumps(body)[:400]}")
                text=body["choices"][0]["message"]["content"]; usage=body.get("usage",{})
                parsed=parse_json_from_text(text)
                return {"forecast_eps":parsed.get("forecast_eps"),"forecast_sales":parsed.get("forecast_sales"),
                    "forecast_profit_margin":parsed.get("forecast_profit_margin"),
                    "confidence":parsed.get("confidence"),
                    "prompt_tokens":usage.get("prompt_tokens",0),"completion_tokens":usage.get("completion_tokens",0),
                    "total_tokens":usage.get("total_tokens",0)}
        raise RuntimeError("OpenAI: max retries (429)")

async def call_anthropic(aio_session,sem,settings,sys_p,usr_p):
    async with sem:
        payload={"model":settings.anthropic_model,"max_tokens":settings.anthropic_max_tokens,
            "temperature":settings.anthropic_temperature,"top_k":30,"system":sys_p,
            "messages":[{"role":"user","content":usr_p}]}
        headers={"x-api-key":settings.anthropic_api_key,"anthropic-version":"2023-06-01","Content-Type":"application/json"}
        for attempt in range(5):
            async with aio_session.post("https://api.anthropic.com/v1/messages",json=payload,headers=headers,timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status==429: await asyncio.sleep(min(2**attempt*2,60)); continue
                body=await resp.json()
                if resp.status>=400: raise RuntimeError(f"Anthropic {resp.status}: {json.dumps(body)[:400]}")
                text="".join(b.get("text","") for b in body.get("content",[]) if b.get("type")=="text")
                usage=body.get("usage",{})
                parsed=parse_json_from_text(text)
                return {"forecast_eps":parsed.get("forecast_eps"),"forecast_sales":parsed.get("forecast_sales"),
                    "forecast_profit_margin":parsed.get("forecast_profit_margin"),
                    "confidence":parsed.get("confidence"),
                    "prompt_tokens":usage.get("input_tokens",0),"completion_tokens":usage.get("output_tokens",0),
                    "total_tokens":usage.get("input_tokens",0)+usage.get("output_tokens",0)}
        raise RuntimeError("Anthropic: max retries (429)")

async def call_gemini(aio_session,sem,settings,sys_p,usr_p):
    async with sem:
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        payload={"system_instruction":{"parts":[{"text":sys_p}]},"contents":[{"parts":[{"text":usr_p}]}],
            "generationConfig":{"temperature":settings.gemini_temperature,"maxOutputTokens":settings.gemini_max_output_tokens,"responseMimeType":"application/json"}}
        for attempt in range(5):
            async with aio_session.post(url,json=payload,timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status==429: await asyncio.sleep(min(2**attempt*2,60)); continue
                body=await resp.json()
                if resp.status>=400: raise RuntimeError(f"Gemini {resp.status}: {json.dumps(body)[:400]}")
                text="".join(p.get("text","") for c in body.get("candidates",[]) for p in c.get("content",{}).get("parts",[]))
                um=body.get("usageMetadata",{})
                parsed=parse_json_from_text(text)
                return {"forecast_eps":parsed.get("forecast_eps"),"forecast_sales":parsed.get("forecast_sales"),
                    "forecast_profit_margin":parsed.get("forecast_profit_margin"),
                    "confidence":parsed.get("confidence"),
                    "prompt_tokens":um.get("promptTokenCount",0),"completion_tokens":um.get("candidatesTokenCount",0),
                    "total_tokens":um.get("totalTokenCount",0)}
        raise RuntimeError("Gemini: max retries (429)")

# ── Single draw task ──
async def run_single_draw(aio_session,sem,settings,provider,model_name,record,forecast_period,draw_number,run_id,run_date_pt):
    sys_p,usr_p,_=build_prompts(record,forecast_period,settings)
    # Add draw-specific variation to break deterministic outputs (especially Claude)
    usr_p+=f"\n\n[Independent simulation draw #{draw_number}. Vary your assumptions slightly — explore a different plausible scenario each time.]"
    base={"run_id":run_id,"run_date_pt":run_date_pt,"forecast_period":forecast_period,
        "company_id":record["company_id"],"ticker":record.get("ticker",""),
        "company_name":record.get("company_name",""),"currency":record.get("currency",""),
        "model_provider":provider,"model_name":model_name,"draw_number":draw_number}
    try:
        if provider=="openai": result=await call_openai(aio_session,sem,settings,sys_p,usr_p)
        elif provider=="anthropic": result=await call_anthropic(aio_session,sem,settings,sys_p,usr_p)
        elif provider=="gemini": result=await call_gemini(aio_session,sem,settings,sys_p,usr_p)
        else: raise ValueError(f"Unknown: {provider}")
        return {**base,**result,"status":"ok","error_message":""}
    except Exception as exc:
        return {**base,"forecast_eps":None,"forecast_sales":None,"forecast_profit_margin":None,
            "confidence":None,"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
            "status":"error","error_message":str(exc)[:500]}

# ── Distribution computation ──
def percentile(data, p):
    if not data: return 0.0
    s=sorted(data); k=(len(s)-1)*(p/100.0); f=int(k); c=f+1
    if c>=len(s): return s[-1]
    return s[f]+(k-f)*(s[c]-s[f])

def calc_skewness(data):
    n=len(data)
    if n<3: return None
    mean=statistics.mean(data); std=statistics.stdev(data)
    if std<1e-12: return 0.0
    return round(sum((x-mean)**3 for x in data)/n/(std**3),6)

def calc_kurtosis(data):
    n=len(data)
    if n<4: return None
    mean=statistics.mean(data); std=statistics.stdev(data)
    if std<1e-12: return 0.0
    return round(sum((x-mean)**4 for x in data)/n/(std**4)-3.0,6)

def compute_distribution(rows, run_id, forecast_period):
    groups=defaultdict(list); combined=defaultdict(list)
    for r in rows:
        if r.get("status")!="ok": continue
        groups[(r["company_id"],r["model_provider"])].append(r)
        combined[r["company_id"]].append(r)
    summaries=[]
    def safe_stats(vals, prefix):
        keys=["mean","median","std","skewness","kurtosis","min","max","p5","p10","p25","p75","p90","p95"]
        if not vals: return {f"{prefix}_{k}":None for k in keys}
        return {
            f"{prefix}_mean":round(statistics.mean(vals),6),
            f"{prefix}_median":round(statistics.median(vals),6),
            f"{prefix}_std":round(statistics.stdev(vals),6) if len(vals)>1 else 0.0,
            f"{prefix}_skewness":calc_skewness(vals),
            f"{prefix}_kurtosis":calc_kurtosis(vals),
            f"{prefix}_min":round(min(vals),6), f"{prefix}_max":round(max(vals),6),
            f"{prefix}_p5":round(percentile(vals,5),6),
            f"{prefix}_p10":round(percentile(vals,10),6),
            f"{prefix}_p25":round(percentile(vals,25),6),
            f"{prefix}_p75":round(percentile(vals,75),6),
            f"{prefix}_p90":round(percentile(vals,90),6),
            f"{prefix}_p95":round(percentile(vals,95),6),
        }
    def make_summary(company_rows, provider, model):
        eps_v=[r["forecast_eps"] for r in company_rows if r.get("forecast_eps") is not None]
        sales_v=[r["forecast_sales"] for r in company_rows if r.get("forecast_sales") is not None]
        pm_v=[r["forecast_profit_margin"] for r in company_rows if r.get("forecast_profit_margin") is not None]
        conf_v=[r["confidence"] for r in company_rows if r.get("confidence") is not None]
        f=company_rows[0]
        base={"run_id":run_id,"forecast_period":forecast_period,
            "company_id":f["company_id"],"ticker":f["ticker"],
            "company_name":f["company_name"],"currency":f["currency"],
            "model_provider":provider,"model_name":model,"n_draws_ok":len(company_rows)}
        base.update(safe_stats(eps_v,"eps"))
        base.update(safe_stats(sales_v,"sales"))
        base.update(safe_stats(pm_v,"pm"))
        base["confidence_mean"]=round(statistics.mean(conf_v),4) if conf_v else None
        return base
    for (cid,prov),rows_g in sorted(groups.items()):
        summaries.append(make_summary(rows_g,prov,rows_g[0]["model_name"]))
    for cid,rows_g in sorted(combined.items()):
        summaries.append(make_summary(rows_g,"combined","all_models"))
    return summaries

# ── Save helpers ──
def save_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def save_csv(path, rows, fieldnames):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
        w.writeheader()
        for row in rows: w.writerow({k:row.get(k,"") for k in fieldnames})

# ── Google Drive ──
def authorize_drive(settings):
    creds=None
    if settings.google_drive_token_file.exists():
        creds=Credentials.from_authorized_user_file(str(settings.google_drive_token_file),DRIVE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token: creds.refresh(GoogleRequest())
        else:
            if not settings.google_drive_credentials_file.exists(): raise FileNotFoundError(f"credentials.json not found")
            flow=InstalledAppFlow.from_client_secrets_file(str(settings.google_drive_credentials_file),DRIVE_SCOPES)
            creds=flow.run_local_server(port=0)
        settings.google_drive_token_file.write_text(creds.to_json(),encoding="utf-8")
    return google_build("drive","v3",credentials=creds)

def escape_dq(t): return t.replace("'","\\'")

def ensure_drive_folder(service,name,parent_id=None):
    q=["mimeType = 'application/vnd.google-apps.folder'",f"name = '{escape_dq(name)}'","trashed = false"]
    if parent_id: q.append(f"'{parent_id}' in parents")
    r=service.files().list(q=" and ".join(q),spaces="drive",fields="files(id, name)").execute()
    files=r.get("files",[])
    if files: return files[0]["id"]
    meta={"name":name,"mimeType":"application/vnd.google-apps.folder"}
    if parent_id: meta["parents"]=[parent_id]
    return service.files().create(body=meta,fields="id").execute()["id"]

def upload_file_to_drive(service,local_path,parent_id):
    mt,_=mimetypes.guess_type(str(local_path))
    media=MediaFileUpload(str(local_path),mimetype=mt or "application/octet-stream",resumable=True)
    return service.files().create(body={"name":local_path.name,"parents":[parent_id]},media_body=media,fields="id, name, webViewLink").execute()

def upload_run_folder_to_drive(service,settings,run_dir,run_dt_pt):
    root_id=settings.google_drive_root_folder_id.strip()
    if not root_id: root_id=ensure_drive_folder(service,settings.google_drive_root_folder_name)
    yid=ensure_drive_folder(service,str(run_dt_pt.year),root_id)
    mid=ensure_drive_folder(service,f"{run_dt_pt.month:02d}",yid)
    rid=ensure_drive_folder(service,run_dir.name,mid)
    uploaded=[]
    for p in sorted(run_dir.iterdir()):
        if p.is_file(): uploaded.append(upload_file_to_drive(service,p,rid))
    return {"drive_root_folder_id":root_id,"drive_run_folder_id":rid,"uploaded_files":uploaded}

# ── Main pipeline ──
import hashlib

def compute_data_hash(records):
    """Deterministic hash of normalized records, ignoring raw_meta."""
    clean=[{k:v for k,v in r.items() if k!="raw_meta"} for r in records]
    blob=json.dumps(clean,sort_keys=True,default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def find_previous_run_hash(output_dir, current_run_id):
    """Look through past run folders to find the most recent data hash."""
    if not output_dir.exists(): return None,None
    runs=sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name!=current_run_id],reverse=True)
    for r in runs:
        hf=r/"data_hash.txt"
        if hf.exists():
            try: return hf.read_text().strip(),r.name
            except Exception: continue
    return None,None

async def run_pipeline_async(settings, skip_drive_upload=False, authorize_drive_only=False):
    tz=ZoneInfo(settings.timezone_name); run_dt=datetime.now(tz)
    run_id=run_dt.strftime("%Y%m%d_%H%M%S_%Z")
    run_date_pt=run_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    date_prefix=run_dt.strftime("%Y%m%d")
    fp=settings.forecast_period or infer_next_quarter_label(run_dt)
    run_dir=settings.local_output_dir/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    setup_logging(run_dir/"pipeline.log")
    logging.info("="*60)
    logging.info("  MULTI-MODEL DISTRIBUTION FORECAST PIPELINE")
    logging.info("="*60)
    logging.info("Run ID: %s | Period: %s | Draws/firm/model: %d",run_id,fp,settings.n_draws)
    if authorize_drive_only:
        svc=authorize_drive(settings)
        save_json(run_dir/"drive_auth_ok.json",svc.about().get(fields="user, storageQuota").execute())
        logging.info("Drive auth done."); return run_dir
    payload=fetch_source_payload_sync(settings,run_dt)
    if settings.source_api_save_raw: save_json(run_dir/"raw_api_payload.json",payload)
    records=normalize_records(payload,settings)
    if not records: raise RuntimeError("No usable company records. Check raw_api_payload.json in output folder.")
    save_json(run_dir/"normalized_records.json",records)

    # ── Check if data is identical to previous run ──
    current_hash=compute_data_hash(records)
    (run_dir/"data_hash.txt").write_text(current_hash,encoding="utf-8")
    prev_hash,prev_run=find_previous_run_hash(settings.local_output_dir,run_id)
    if prev_hash==current_hash and prev_hash is not None:
        logging.info("="*60)
        logging.info("  DATA UNCHANGED since previous run (%s)",prev_run)
        logging.info("  Skipping AI forecasts to save API costs.")
        logging.info("="*60)
        summary={"run_id":run_id,"run_date_pt":run_date_pt,"forecast_period":fp,
            "status":"skipped_data_unchanged","previous_run":prev_run,
            "data_hash":current_hash,"companies":len(records)}
        save_json(run_dir/"run_summary.json",summary)
        # Still upload the tiny run folder so you have a record
        if not skip_drive_upload:
            try:
                svc=authorize_drive(settings)
                di=upload_run_folder_to_drive(svc,settings,run_dir,run_dt)
                summary["drive_info"]=di; save_json(run_dir/"run_summary.json",summary)
                logging.info("Drive upload complete (skip-marker only).")
            except Exception as exc:
                logging.warning("Drive upload failed: %s",exc)
        return run_dir

    providers=[]
    if settings.enable_openai and settings.openai_api_key:
        providers.append(("openai",settings.openai_model,asyncio.Semaphore(settings.openai_concurrency)))
    if settings.enable_anthropic and settings.anthropic_api_key:
        providers.append(("anthropic",settings.anthropic_model,asyncio.Semaphore(settings.anthropic_concurrency)))
    if settings.enable_gemini and settings.gemini_api_key:
        providers.append(("gemini",settings.gemini_model,asyncio.Semaphore(settings.gemini_concurrency)))
    if not providers: raise RuntimeError("No providers enabled.")
    total=len(records)*len(providers)*settings.n_draws
    logging.info("-"*60)
    for pn,pm,_ in providers: logging.info("  %-12s -> %s",pn,pm)
    logging.info("-"*60)
    logging.info("Total: %d firms x %d providers x %d draws = %d tasks",len(records),len(providers),settings.n_draws,total)
    tasks=[]
    conn=aiohttp.TCPConnector(limit=200,limit_per_host=80)
    async with aiohttp.ClientSession(connector=conn) as aio:
        for rec in records:
            for pn,pm,sem in providers:
                for d in range(1,settings.n_draws+1):
                    tasks.append(run_single_draw(aio,sem,settings,pn,pm,rec,fp,d,run_id,run_date_pt))
        logging.info("Dispatching %d async tasks...",len(tasks))
        all_rows=await asyncio.gather(*tasks)
    all_rows=list(all_rows)
    ok=sum(1 for r in all_rows if r.get("status")=="ok"); err=len(all_rows)-ok
    logging.info("Results: %d ok, %d errors",ok,err)
    # Date-prefixed CSV filenames
    save_csv(run_dir/f"{date_prefix}_per_draw_results.csv",all_rows,DRAW_OUTPUT_FIELDS)
    save_json(run_dir/f"{date_prefix}_per_draw_results.json",all_rows)
    failures=[r for r in all_rows if r.get("status")!="ok"]
    if failures: save_json(run_dir/f"{date_prefix}_failures.json",failures)
    dist=compute_distribution(all_rows,run_id,fp)
    save_csv(run_dir/f"{date_prefix}_distribution_summary.csv",dist,DIST_SUMMARY_FIELDS)
    save_json(run_dir/f"{date_prefix}_distribution_summary.json",dist)
    logging.info("Distribution summaries: %d rows",len(dist))
    summary={"run_id":run_id,"run_date_pt":run_date_pt,"forecast_period":fp,
        "n_draws_per_firm_per_model":settings.n_draws,
        "providers":[{"name":p[0],"model":p[1]} for p in providers],
        "companies":len(records),"total_tasks":total,"ok":ok,"errors":err,
        "data_hash":current_hash}
    save_json(run_dir/"run_summary.json",summary)
    if not skip_drive_upload:
        logging.info("Uploading to Google Drive...")
        svc=authorize_drive(settings)
        di=upload_run_folder_to_drive(svc,settings,run_dir,run_dt)
        summary["drive_info"]=di; save_json(run_dir/"run_summary.json",summary)
        logging.info("Drive upload complete.")
    else: logging.info("Skipping Drive upload.")
    logging.info("="*60)
    logging.info("  DONE.  Output: %s",run_dir)
    logging.info("="*60)
    return run_dir

def parse_args():
    p=argparse.ArgumentParser(description="Multi-model distribution forecast pipeline")
    p.add_argument("--run-now",action="store_true")
    p.add_argument("--authorize-drive-only",action="store_true")
    p.add_argument("--skip-drive-upload",action="store_true")
    p.add_argument("--n-draws",type=int,default=None)
    return p.parse_args()

def main():
    try:
        settings=load_settings(); args=parse_args()
        if args.n_draws is not None: settings.n_draws=args.n_draws
        run_dir=asyncio.run(run_pipeline_async(settings,skip_drive_upload=args.skip_drive_upload,authorize_drive_only=args.authorize_drive_only))
        print(f"DONE: {run_dir}"); return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s",exc); print(f"ERROR: {exc}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
