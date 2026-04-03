# Bug: LLM Judge "Run Evaluation" button does nothing

## Symptom

In the Streamlit app, after a successful pipeline run (all 7 agents: Profiler, Analyst, Critic, Uncertainty, Visualizer, Reporter, Memory — all green/success), clicking the **"▶ Run Evaluation"** button in the **🔍 LLM Judge** tab causes the page to reload but the evaluation never executes. No error is displayed, no logs are printed in the terminal.

## Architecture

- **Framework**: Streamlit 1.45+ with `streamlit-flow-component` (React Flow)
- **LLM**: Gemini 3.1 Pro Preview via `langchain-google-genai`
- **Pipeline**: LangGraph StateGraph, progressive streaming
- **The button**: Located in `_render_llm_judge_tab()` inside `app/main.py`

## The pipeline diagram uses `streamlit_flow`

The app renders a React Flow pipeline diagram (`streamlit_flow()` from `streamlit-flow-component`) **above** the result tabs. This custom component is known to trigger extra Streamlit reruns when it mounts/renders.

## What happens when the button is clicked

1. `st.button("▶ Run Evaluation")` returns `True`
2. Code sets `st.session_state["_run_judge_requested"] = True`
3. `st.rerun()` is called
4. On the new run cycle, **before** reaching the judge execution code, `streamlit_flow()` renders and triggers **another rerun**
5. On this third run, `_run_judge_requested` has already been consumed/lost
6. The judge never executes

## Current code structure

### `main()` (line ~1467 in app/main.py):
```python
def main() -> None:
    st.set_page_config(...)

    # Attempt 3: Run judge at very top of main()
    if st.session_state.get("_run_judge_requested"):
        st.session_state.pop("_run_judge_requested", None)
        result = st.session_state.get("analysis_result")
        if result:
            try:
                _run_llm_judge(result)
            except Exception as exc:
                st.session_state["_judge_error"] = str(exc)

    st.markdown(_CUSTOM_CSS, ...)
    # ... header, sidebar, file upload ...
    # ... React Flow diagram (streamlit_flow) renders here ...
    # ... result tabs including LLM Judge tab ...
```

### `_render_llm_judge_tab()` (line ~830):
```python
def _render_llm_judge_tab(result, key_suffix=""):
    st.markdown("### 🔍 LLM-as-Judge Quality Evaluation")
    pipeline_eval = st.session_state.get("pipeline_eval")

    judge_error = st.session_state.pop("_judge_error", None)
    if judge_error:
        st.error(f"LLM Judge failed: {judge_error}")

    if pipeline_eval is None:
        if st.button("▶ Run Evaluation", type="primary", key=f"run_eval{key_suffix}"):
            st.session_state["_run_judge_requested"] = True
            st.rerun()
        st.info("Click Run Evaluation to score...")
        return
    # ... render results ...
```

### `_run_llm_judge()` (line ~791):
```python
def _run_llm_judge(result):
    from evaluation.llm_judge import EvaluationAgent
    from evaluation.schemas import PipelineEvaluation

    agent = EvaluationAgent()
    pipeline_eval = PipelineEvaluation()

    if result.get("profile_markdown"):
        with st.spinner("🔍 Evaluating profiler report…"):
            pipeline_eval.profiler_eval = agent.evaluate_profiler(...)
    if result.get("insights"):
        with st.spinner("🔍 Evaluating analyst insights…"):
            pipeline_eval.analyst_eval = agent.evaluate_analyst(...)
    if result.get("report_markdown"):
        with st.spinner("🔍 Evaluating final report…"):
            pipeline_eval.reporter_eval = agent.evaluate_reporter(...)

    st.session_state["pipeline_eval"] = pipeline_eval
```

### The React Flow diagram (`diagnostics/renderer.py` line ~176):
```python
streamlit_flow(
    key=f"pipeline_flow{key_suffix}",
    state=flow_state,
    height=320,
    fit_view=True,
    show_controls=False,
    show_minimap=False,
    allow_new_edges=False,
    pan_on_drag=False,
    allow_zoom=False,
    layout=ManualLayout(),
    hide_watermark=True,
)
```

## What we tried (all failed)

### Attempt 1: Direct execution in button callback
```python
if st.button("▶ Run Evaluation"):
    _run_llm_judge(result)
    st.rerun()
```
**Result**: Page reloads, nothing happens. The button is inside a `st.tabs()` context which gets destroyed on rerun.

### Attempt 2: session_state flag + execute inside tab
```python
if st.session_state.get("_run_judge_requested"):
    st.session_state.pop("_run_judge_requested", None)
    _run_llm_judge(result)
    st.rerun()
```
**Result**: Same — the flag is consumed by React Flow's extra rerun before execution.

### Attempt 3: session_state flag + execute at top of main()
Moved the judge execution to the very beginning of `main()`, before any component renders.
**Result**: Still doesn't work — possibly `st.set_page_config()` triggers its own behavior, or the flag is lost between rerun cycles triggered by the React Flow component.

### Attempt 4: Execute in `_render_result_tabs()` before tab creation
**Result**: Same issue — React Flow diagram renders before tabs.

## Key suspicion

The `streamlit_flow` custom component causes phantom reruns. When the button sets a session_state flag and calls `st.rerun()`, the sequence is:
1. Rerun starts
2. `streamlit_flow` component sends a message back to Python, causing ANOTHER rerun
3. The flag is popped on rerun #1 but the judge blocks on LLM I/O
4. Rerun #2 interrupts/resets everything

## Possible solutions to try

1. **Use `st.form` with `st.form_submit_button`** to prevent the button from triggering a rerun — instead batch the action
2. **Use `on_click` callback** instead of checking `st.button()` return value — callbacks execute before the page rerenders
3. **Replace `streamlit_flow` with static HTML/SVG** for the pipeline diagram to eliminate phantom reruns
4. **Use `st.fragment`** (Streamlit 1.33+) to isolate the judge tab from the rest of the page reruns
5. **Use a threading approach** — start judge in a background thread, poll for completion
6. **Disable `streamlit_flow` interactivity** or wrap it so it doesn't trigger reruns

## Environment

- Python 3.12.3
- Streamlit (latest)
- streamlit-flow-component
- langchain-google-genai (Gemini 3.1 Pro Preview)
- Linux (WSL2), 7.6 GB RAM
- LLM calls take 20-40 seconds each (3 calls total for judge)
