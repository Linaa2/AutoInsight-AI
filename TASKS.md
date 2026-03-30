For everyone: **Ollama integration for local LLM testing**

- **P0** (Infrastructure) - 29/03/2026 - Amine:
    - project structure
    - CI/CD pipelines
    - pre-commit hooks
    - uvicorn + streamlit setup
    - Docker setup for local deployment (optional)

- **P1** (Profiler + UI + Test) - 29/03/2026 - Younes:
    - data loading (different formats csv, excel, parquet)
    - deterministic profiling using pandas: shape, column types, missing values, duplicates, descriptive stats, value distributions, sample rows
        - output schema for profile (e.g., JSON structure)
        ```json
        {
          "shape": [1000, 12],
          "columns": {...},
          "missing": {...},
          "stats": {...},
          "samples": [...]
        }
        ```
        - (optional) ensure profiling is efficient and can handle large datasets (e.g., using chunking or sampling if necessary)
    - profiler agent: generates structured markdown description of the profile
        - explain structure, highlight data quality issues, etc.
        - ensure markdown is well-formatted and easy to read
        - consider adding visual elements (e.g., emojis, bullet points) to enhance readability
        - prompts should encourage the agent to provide insights, not just raw statistics
        - ensure the agent can handle different types of datasets and adapt its description accordingly
        - consider adding a section for "Key Takeaways" where the agent summarizes the most important insights from the profile
        - sections: overview, data quality, column types, distributions, etc.
    - UI for uploading dataset and displaying profile, with focus on clarity and usability

- **P2** (Analyst + UI + Test) - 29/03/2026 - Lina: 
    - insight generation agent that takes profile as input and generates insights in markdown format
        - prompts should encourage the agent to provide actionable insights, not just descriptive statistics
        - define a set of insight categories (e.g., trends, anomalies, correlations) and ensure the agent covers them
        - ensure insights are concise, clear, and well-structured
        - input to the agent should include both the profile and sample rows to provide context for generating insights
        - output format for insights (e.g., JSON structure with title, description, etc.)
        ```json
        {
          "insights": [
            {
              "title": "...",
              "description": "..."
            },
            ...
          ]
        }
        ```
    - insights display in UI (*!! wait for p1!!*)
        - add a separate page or section for insights, with clear formatting and visual hierarchy
        - consider adding visual elements (e.g., icons, colors) to differentiate between different types of insights (e.g., trends vs anomalies)
            - ensure the UI can handle multiple insights and that they are organized in a clear and intuitive manner
            - consider adding a feature for users to provide feedback on insights (e.g., thumbs up/down) to help improve future insight generation
            - ensure the UI can handle different screen sizes and that insights are displayed in a responsive manner
    - (optional) data validation layer: checks if insights are consistent with data (no hallucinations): encourage the agent to consider data quality issues when generating insights and to mention any limitations or uncertainties in the insights
    
- **P3** (Visualizer + UI + Test) - 29/03/2026 - Amine:
    - visualizer agent: generates chart specs (title, type, code) based on insights + profile
        - prompts should encourage the agent to generate charts that are relevant to the insights and that effectively communicate the underlying data patterns
        - define a set of chart types (e.g., bar, line, scatter) and ensure the agent can choose the most appropriate type based on the insight and data
        - ensure generated code is correct and can be executed without errors
        - consider adding a feedback loop where the agent can learn from user interactions with the charts (e.g., if a user frequently modifies a certain type of chart, the agent could learn to generate it more often)
        - ensure the agent can handle different types of datasets and adapt its chart generation accordingly
        - consider adding a section for "Chart Explanation" where the agent explains why it chose a particular chart type and what the user should look for in the chart
    - chart rendering in UI
         - ensure charts are rendered correctly and are visually appealing
         - add interactivity to charts (e.g., tooltips, zooming) to enhance user experience
         - consider adding a feature for users to modify the generated charts (e.g., change chart type, adjust axes) and have the agent learn from these modifications to improve future chart generation
         - ensure the UI can handle multiple charts and that they are organized in a clear and intuitive manner
         - consider adding a feature for users to save or export charts for use in reports or presentations
         - ensure the UI can handle different screen sizes and that charts are responsive
    - executor: executes chart code and returns results
        - ensure code execution is safe and does not pose security risks (e.g., by using a sandboxed environment)
        - ensure code execution is efficient and can handle complex charts without significant delays
    - UI

- **P4** (Orchestrator LangGraph + UI + Test):
    - orchestrator agent: manages workflow between profiler, analyst, visualizer based on user query
    - LangGraph implementation of the workflow
    - LangFuse integration for tracing and debugging

- **P5** (ChromaDB + RAG + UI + Test):
    - ChromaDB integration for storing profiles, insights, critiques, etc.
    - RAG agent: retrieves relevant past insights, critiques, etc. to inform current analysis
    - UI for browsing past analyses and their critiques

- **P6** (Text-to-Code + UI + Test):
    - text-to-code agent: generates pandas code from user query + profile
    - sandbox: safely executes code and returns results
    - UI for Q&A

- **P7** (Evaluation + UI + Test):
    - evaluation agent (LLM-as-judge): evaluates quality of insights, visualizations, code, etc. based on criteria like correctness, relevance, clarity
    - UI for displaying evaluations and critiques

- **P8** (Critic + Uncertainty + Report + Test):
    - critic agent: critiques insights, identifies weaknesses, suggests alternatives
    - uncertainty estimator: assigns confidence scores to insights based on data quality, critique, etc.
    - enhanced report integrating insights + critique + uncertainty