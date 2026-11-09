"""Streamlit demo UI:  streamlit run app.py"""
import streamlit as st

from detector import check_answer

st.set_page_config(page_title="Stale-Answer Detector", page_icon="🕰️", layout="wide")
st.title("🕰️ Stale-Answer Detector")
st.caption("Paste any AI answer. We verify its time-sensitive claims against the live web.")

EXAMPLES = {
    "Outdated CEO": ("Who is the CEO of Twitter?",
                     "Jack Dorsey is the CEO of Twitter and he also runs Square."),
    "Old Python version": ("What is the latest stable version of Python?",
                           "The latest stable version of Python is 3.8, which brings the walrus operator."),
    "Old PM": ("Who is the UK Prime Minister?",
               "The current UK Prime Minister is Boris Johnson, leader of the Conservative Party."),
    "Old iPhone": ("What is the newest iPhone?",
                   "The newest iPhone is the iPhone 12, released with 5G support."),
}

if "q" not in st.session_state:
    st.session_state.q, st.session_state.a = "", ""

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (name, (q, a)) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.q, st.session_state.a = q, a

left, right = st.columns(2)
with left:
    q = st.text_input("Question", value=st.session_state.q)
    a = st.text_area("AI answer to check", value=st.session_state.a, height=180)
    run = st.button("Check freshness", type="primary", disabled=not (q and a))

BADGE = {"fresh": "🟢 FRESH", "stale": "🔴 STALE", "uncertain": "🟡 UNCERTAIN"}

with right:
    if run:
        with st.spinner("Extracting claims and checking the live web..."):
            try:
                res = check_answer(q, a)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.subheader(BADGE[res["verdict"]])
        m1, m2, m3 = st.columns(3)
        m1.metric("Freshness score", f"{int(res['freshness_score'] * 100)}%")
        m2.metric("Claims checked", len(res["claims"]))
        m3.metric("Latency", f"{res['latency_s']}s")

        if res.get("note"):
            st.info(res["note"])

        for c in res["claims"]:
            with st.expander(f"{BADGE[c['verdict']]}  |  {c['claim']}", expanded=c["verdict"] == "stale"):
                st.write(c["reason"])
                if c.get("corrected_claim"):
                    st.success(f"Correction: {c['corrected_claim']}")
                st.caption(f"Confidence: {c['confidence']:.2f}  |  Volatility: {c.get('volatility', '-')}")
                for url in c["source_urls"][:3]:
                    st.markdown(f"- [{url}]({url})")

        if res["verdict"] == "stale":
            st.markdown("### ✅ Corrected answer")
            st.success(res["corrected_answer"])
    else:
        st.info("Results will appear here.")
