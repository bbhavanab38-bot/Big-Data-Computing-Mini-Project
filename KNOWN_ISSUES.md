# Known issues found in a prior review (verify each; fix in the phase noted)

1. Data is synthetic, generated from a logistic formula. Best model AUC 0.729 vs
   theoretical oracle ceiling 0.748. Logistic Regression beats XGBoost because of this.
   Do not tune models. Label data as synthetic everywhere. (all phases)
2. class_weight="balanced" makes scores uncalibrated: mean score 0.436 vs actual churn rate
   0.164. The gauge, the 30/70 risk bands and the API show scores as percentages. (Phase 2)
3. reports/powerbi/powerbi_dataset.csv scores all 10,000 rows including 8,000 training
   rows (train AUC 0.750 vs test 0.729). The dashboard and ROI simulator use it. The
   headline "optimal N" is unstable (1,464 on all rows vs about 480 on test-only scaled).
   (Phase 2)
4. Flask API: Age="abc" and null return HTTP 500; Age=-5 / Balance=-1e9 are accepted and
   return "Likely To Stay 0%"; unknown Geography is silently accepted; probability is
   returned as a string; no auth, no model version; binds 0.0.0.0. (Phase 2)
5. streamlit and plotly are imported but missing from requirements.txt. (Phase 2)
6. explain_customer_prediction only works for models with coef_ (silently empty for tree
   models). Dashboard text hardcodes "Logistic Regression". (Phase 4)
7. dashboard/core/segments.py hardcodes segment names to one run's K-Means cluster IDs and
   no Segment column exists in customer-level data. (Phase 2)
8. Gender is a model input; Spark output shows Germany/Female 27.3% churn vs France/Male
   13.1%. No fairness audit exists. (Phase 5)
9. 4 of 5 engineered features have near-zero permutation importance (DigitalUsageScore is
   negative). Report this honestly; do not hide it. (Phase 2)
10. There is no date column, so no monthly trend charts or date filters until the
    simulated event stream produces timestamps. (Phase 6)
11. There is no treatment/control data. The decision engine must use a clearly labelled
    simulator. (Phase 4)
12. Spark Decision Tree AUC is 0.48 (below chance) in reports/spark; unexplained.
    Investigate and report; do not hide it. (Phase 2)
13. Spark jobs write with errorifexists and cannot be re-run without deleting outputs.
    (Phase 2, document or add an opt-in overwrite flag)
14. README describes Power BI as the dashboard and omits the Streamlit app; docs and logs
    contain hardcoded /Users/vikas paths; stray files yarn_test.py and powerbi_error.log;
    model pickles carry no library-version metadata. (Phase 2 and Phase 7)
