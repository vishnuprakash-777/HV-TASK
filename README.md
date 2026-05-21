# Patient Longitudinal Health Intelligence

A proof of concept that turns isolated blood-test records into a coherent health
story for each patient. Instead of reading every lab report on its own, it looks
across all of a patient's visits to work out whether each marker is improving,
worsening, or holding steady, how their overall health burden is trending, and
produces a short plain-language summary for a clinical care coordinator.

Built as a take-home task on a dataset of 1,000 patients with repeat blood tests
spanning 2019 to 2026.

## Dataset

The dataset (`patient_health_data.csv`) is **not included in this repository**,
as it is confidential. To run the notebook or the dashboard, place your own copy
of `patient_health_data.csv` in the project root, alongside the other files.

## What's in this repo

| File | Description |
|---|---|
| `analysis.ipynb` | The main notebook. End-to-end analysis with all outputs. |
| `patient_summary.csv` | The output CSV, one row per patient (LLM summaries from model 1). |
| `patient_summary_2.csv` | Same output, with LLM summaries from a second model, for comparison. |
| `dashboard.py` | A Streamlit dashboard for browsing the results. |

## The notebook (analysis.ipynb)

The notebook is organised into five sections:

1. **Data loading and exploration** — loads the dataset and checks data quality
   (missing values, visits per patient, parameter coverage). It also flags that
   reference ranges vary between patients (some are gender-specific), so the
   per-row ranges in the data are used rather than fixed defaults.

2. **Longitudinal trend signals** — for every patient-parameter pair it computes
   direction of change, magnitude, slope (via linear regression), normal-boundary
   crossings, and abnormal-visit counts. These feed a rule-based classifier that
   assigns one of 9 trajectory labels (`steadily_worsening`, `approaching_risk`,
   `recovering`, `stable_normal`, and so on). Series with too few visits are
   labelled `insufficient_data`.

3. **Cross-parameter burden analysis** — counts abnormal markers per visit, tracks
   a health burden score over time, and finds which parameter pairs most often go
   abnormal together and which one tends to go abnormal first.

4. **LLM-generated health summaries** — builds a plain-language context for each
   patient and uses an LLM to write a 2 to 4 sentence summary written for a
   clinical care coordinator.

5. **Findings** — answers the task questions: the trajectory distribution, which
   parameters drive the urgent cases, the strongest co-occurring pairs, and the
   most consistent temporal orderings.

## Output CSVs

`patient_summary.csv` has one row per patient:

| Column | Description |
|---|---|
| `patient_id` | Patient identifier |
| `gender` | Gender |
| `age_band` | Age band |
| `total_visits` | Number of distinct test visits |
| `param_trends` | JSON of `{parameter: trajectory_label}` |
| `current_abnormal_params` | Parameters currently outside the normal range |
| `burden_trend` | Overall direction: improving / worsening / stable |
| `worst_trajectory` | The most urgent trajectory across all parameters |
| `llm_summary` | The LLM-generated plain-language summary |

`patient_summary_2.csv` is the same file, but the `llm_summary` column was
generated with a different LLM model. The two files are included so the summary
outputs of the two models can be compared side by side.

The LLM step uses [Groq](https://groq.com) to run Llama models. The summaries
were generated with two different models — see the notebook for the exact model
names — which is why there are two CSVs.

## Dashboard (dashboard.py)

A Streamlit dashboard for browsing the results, with three views:

- **Population overview** — cohort-level stats and the trajectory distribution.
- **Risk ranking** — all patients sorted by urgency, with filters and CSV export.
- **Patient explorer** — one patient's metrics, LLM summary, health-burden chart,
  and a trend chart for each marker.

Run it with:

```bash
pip install streamlit pandas numpy matplotlib
streamlit run dashboard.py
```

The dashboard needs `patient_summary.csv` and your copy of
`patient_health_data.csv` in the same folder as the script.

## Running the notebook

```bash
pip install pandas numpy matplotlib jupyter openai
jupyter notebook analysis.ipynb
```

Place your copy of `patient_health_data.csv` in the project root first.

The LLM step needs a Groq API key, set as the `GROQ_API_KEY` environment
variable. Without a key the notebook still runs end to end — it falls back to a
deterministic rule-based summary instead of calling the LLM.

## Notes and limitations

- **Dataset** is confidential and is not committed to this repository.
- **LLM coverage**: the free-tier daily token limit on the API was reached
  partway through the run, so not every patient has an LLM-generated summary.
  The remaining patients use the deterministic fallback. The analysis pipeline
  is identical either way.
- **Reference ranges** in the dataset are treated as the ground truth for normal
  vs abnormal, as the task specifies.
- **Trajectory thresholds** (minimum visits, what counts as a meaningful change)
  are exposed as tunable constants in Section 2 of the notebook.
- This is a proof of concept, not production code, and not clinical advice.
