# Project rules (read this file at the start of every task)

Product: Banking Retention Intelligence Platform, an upgrade of the existing project
"Finance and Banking Data Analytics: A YARN Approach to Churn Prediction".

General
- Work on the git branch "upgrade". Commit after each completed phase.
- Inspect before changing. Preserve all working functionality. Do not delete files without
  a stated technical reason. Existing Hadoop/HDFS/Spark/YARN scripts must keep working;
  only fix real bugs in them and report each fix.
- Never replace real ML or real data with dummy values. Never fabricate metrics, charts,
  explanations, API integrations or customer events.
- All data in this project is SYNTHETIC. Every place where data is simulated (customer
  data, treatment/uplift, event streams, campaign responses) must be clearly labelled as
  simulated in the UI and in code. Never present simulated uplift or simulated outcomes as
  real-world measured results.
- Do not claim a feature is complete unless you have run it and its tests. If something
  cannot be run in this environment (Docker, Spark/YARN, network), say so explicitly;
  never hide failures.
- Every button, filter, chart and navigation item must work with actual project data.
  Remove or clearly disable anything that cannot work yet; no placeholder widgets.
- Do not tune or swap models to chase metrics. The data is synthetic and the models are
  already near the theoretical ceiling.
- Keep code modular: business logic in pure-Python modules (no Streamlit imports), UI in
  separate page modules. No duplicate modules for the same purpose.
- Add new dependencies only when needed; list each with a one-line reason in
  requirements.txt comments. Pin versions.
- Security: no secrets in code or git (use .env and .env.example); validate all inputs;
  never load pickle/joblib files from user uploads (uploads are CSV only); models are
  loaded only from the project's own models/ directory.
- Tests: existing tests must keep passing. Add tests for every new module.
- At the end of every phase: stop, report changed files, test results and known gaps,
  append to PROGRESS.md, commit, and WAIT for approval before starting the next phase.
