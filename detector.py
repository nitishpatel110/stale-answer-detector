"""Stale-Answer Detector: core pipeline.

question + answer -> extract volatile claims -> search live web (Tavily)
-> LLM judge per claim -> aggregate verdict + corrected answer.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_CLAIMS = 6          # cap claims per answer to control cost and latency
STALE_CONFIDENCE = 0.6  # below this, a "stale" verdict is downgraded to "uncertain"


# ---------- clients ----------
@lru_cache(maxsize=1)
def _llm() -> OpenAI:
    # Set LLM_BASE_URL to point at Bifrost (or any OpenAI-compatible gateway)
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "none"),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


@lru_cache(maxsize=1)
def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _chat_json(prompt: str, retries: int = 2) -> dict:
    """Call the LLM and robustly parse a JSON object from the reply."""
    last_err = None
    for _ in range(retries + 1):
        try:
            r = _llm().chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Respond with a single valid JSON object only. No prose, no markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = r.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.S)
            return json.loads(match.group(0))
        except Exception as e:  # retry on parse or network errors
            last_err = e
    raise RuntimeError(f"LLM call failed: {last_err}")


# ---------- step 1: extract claims ----------
EXTRACT_PROMPT = """Today is __TODAY__.
Extract the factual claims from the ANSWER that could have become outdated:
prices, versions, office holders, CEOs, records, rankings, laws, availability,
latest releases, sports results, statistics, "current" or "latest" statements.
Skip timeless facts (math, geography, history that cannot change).
For each claim write a specific web search query (include names and the year if useful).

Return JSON: {"claims":[{"claim":"...","search_query":"...","volatility":"high|medium|low"}]}
Return {"claims":[]} if nothing can go stale.

QUESTION: __Q__
ANSWER: __A__"""


def extract_claims(question: str, answer: str) -> list[dict]:
    prompt = (EXTRACT_PROMPT.replace("__TODAY__", date.today().isoformat())
              .replace("__Q__", question).replace("__A__", answer))
    claims = _chat_json(prompt).get("claims", [])
    claims = [c for c in claims if c.get("claim") and c.get("search_query")]
    order = {"high": 0, "medium": 1, "low": 2}
    claims.sort(key=lambda c: order.get(c.get("volatility", "low"), 2))
    return claims[:MAX_CLAIMS]


# ---------- step 2: gather evidence ----------
@lru_cache(maxsize=256)
def _search_cached(query: str) -> tuple:
    res = _tavily().search(query=query, max_results=5, search_depth="advanced")
    return tuple(
        (r.get("title", ""), r.get("url", ""), (r.get("content") or "")[:700])
        for r in res.get("results", [])
    )


def get_evidence(query: str) -> list[dict]:
    try:
        return [{"title": t, "url": u, "content": c} for t, u, c in _search_cached(query)]
    except Exception:
        return []


# ---------- step 3: judge ----------
JUDGE_PROMPT = """Today is __TODAY__.
Decide whether the CLAIM is still true, using ONLY the EVIDENCE (web results fetched just now).

Rules:
- "stale": evidence clearly shows the claim was once true or is now false/superseded.
- "fresh": evidence supports the claim as currently true.
- "uncertain": evidence is missing, weak, off-topic or conflicting. Do not guess.
- Prefer newer, authoritative sources. Give a corrected_claim only when verdict is "stale".

Return JSON:
{"verdict":"fresh|stale|uncertain","reason":"one sentence",
 "corrected_claim":"string or null","source_urls":["..."],"confidence":0.0-1.0}

CLAIM: __CLAIM__
EVIDENCE:
__EVIDENCE__"""


def judge_claim(claim: dict) -> dict:
    evidence = get_evidence(claim["search_query"])
    if not evidence:
        return {**claim, "verdict": "uncertain", "reason": "No evidence found.",
                "corrected_claim": None, "source_urls": [], "confidence": 0.0, "evidence": []}
    ev_text = "\n\n".join(f"[{i+1}] {e['title']} ({e['url']})\n{e['content']}" for i, e in enumerate(evidence))
    prompt = (JUDGE_PROMPT.replace("__TODAY__", date.today().isoformat())
              .replace("__CLAIM__", claim["claim"]).replace("__EVIDENCE__", ev_text))
    try:
        out = _chat_json(prompt)
    except Exception as e:
        out = {"verdict": "uncertain", "reason": f"Judge error: {e}", "corrected_claim": None,
               "source_urls": [], "confidence": 0.0}
    verdict = out.get("verdict", "uncertain")
    conf = float(out.get("confidence") or 0)
    if verdict == "stale" and conf < STALE_CONFIDENCE:
        verdict = "uncertain"  # avoid crying wolf on weak evidence
    return {**claim, "verdict": verdict, "reason": out.get("reason", ""),
            "corrected_claim": out.get("corrected_claim"),
            "source_urls": out.get("source_urls") or [e["url"] for e in evidence[:2]],
            "confidence": conf, "evidence": evidence}


# ---------- step 4: rewrite ----------
REWRITE_PROMPT = """Today is __TODAY__.
Rewrite the ANSWER so it is correct and current, fixing only the outdated claims listed.
Keep the original style and length. Mention the update briefly where natural.

QUESTION: __Q__
ANSWER: __A__
OUTDATED CLAIMS AND CORRECTIONS:
__FIXES__

Return JSON: {"corrected_answer":"..."}"""


def rewrite_answer(question: str, answer: str, stale: list[dict]) -> str:
    fixes = "\n".join(f"- {c['claim']} -> {c.get('corrected_claim') or 'see: ' + c['reason']}" for c in stale)
    prompt = (REWRITE_PROMPT.replace("__TODAY__", date.today().isoformat())
              .replace("__Q__", question).replace("__A__", answer).replace("__FIXES__", fixes))
    try:
        return _chat_json(prompt).get("corrected_answer", answer)
    except Exception:
        return answer


# ---------- main entry ----------
def check_answer(question: str, answer: str) -> dict:
    t0 = time.time()
    claims = extract_claims(question, answer)
    if not claims:
        return {"verdict": "fresh", "freshness_score": 1.0, "claims": [],
                "corrected_answer": answer, "latency_s": round(time.time() - t0, 2),
                "note": "No time-sensitive claims found."}

    with ThreadPoolExecutor(max_workers=min(5, len(claims))) as pool:
        results = list(pool.map(judge_claim, claims))

    stale = [r for r in results if r["verdict"] == "stale"]
    uncertain = [r for r in results if r["verdict"] == "uncertain"]
    fresh = [r for r in results if r["verdict"] == "fresh"]

    verdict = "stale" if stale else ("uncertain" if uncertain else "fresh")
    corrected = rewrite_answer(question, answer, stale) if stale else answer

    return {
        "verdict": verdict,
        "freshness_score": round(len(fresh) / len(results), 2),
        "claims": results,
        "corrected_answer": corrected,
        "latency_s": round(time.time() - t0, 2),
    }
