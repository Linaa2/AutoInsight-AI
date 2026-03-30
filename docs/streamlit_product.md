# AutoInsight-AI — Streamlit Product Guide

## User Experience Flow

1. **Landing** — Welcome message with instructions.
2. **Upload** — Sidebar file uploader (CSV / Excel / Parquet).
3. **Preview** — KPI row (rows, columns, duplicates, missing %, memory) + data expander.
4. **Run** — Single "Run Full Analysis" button invokes the entire pipeline.
5. **Results** — Five tabs display the coordinated output:
   - **Profile** — LLM-interpreted dataset description
   - **Insights** — Structured insight cards or markdown view
   - **Visualizations** — Auto-generated Plotly charts with explanations
   - **Report** — Executive summary with download button
   - **Diagnostics** — Graph execution trace

## Result Sections

### Profile Tab
Displays the markdown report from the Profiler agent (dataset overview,
data quality assessment, column analysis, statistical highlights,
recommendations).

### Insights Tab
Toggle between:
- **Cards** — Priority-colored expanders with observation / hypothesis / recommendation
- **Markdown** — Full categorized markdown output

### Visualizations Tab
Each chart rendered with:
- Title and chart type badge
- Optional explanation
- Plotly interactive figure
- Expandable generated code section

### Report Tab
Full executive report in markdown with a download button (`.md`).

### Diagnostics Tab
Per-node summary metrics row + expandable details:
- Status badge (✅ / ❌ / ⏭️)
- Duration, timestamps
- Keys read / written
- Summary text
- Error details (auto-expanded on failure)

## Sidebar Controls

- **Upload** — File uploader
- **Re-run** — Clears cached results and re-invokes the pipeline

## Architecture Notes

- The app calls `orchestration.graph.run_analysis()` — one function, one invocation.
- All session state is keyed to the uploaded file name; changing files resets state.
- Chart figures are re-executed at render time (figures are not JSON-serializable).
- No business logic lives in the UI layer.
