# Stale-Answer Detector

Checks whether an AI answer contains outdated facts by verifying its time-sensitive claims against live web results (Tavily), then returns a verdict, sources and a corrected answer.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env      # add TAVILY_API_KEY and OPENAI_API_KEY
streamlit run app.py      # demo UI
uvicorn api:app --reload  # optional REST API: POST /check {question, answer}
python eval.py            # benchmark numbers for your pitch
```

## Bifrost (optional)
Start the Bifrost gateway, then set `LLM_BASE_URL` in `.env` to its OpenAI-compatible endpoint
and `LLM_MODEL` to the model name Bifrost expects. All LLM calls will then flow through it.

## Pipeline
extract claims -> Tavily search per claim (parallel) -> LLM judge (fresh/stale/uncertain)
-> aggregate -> rewrite corrected answer.
