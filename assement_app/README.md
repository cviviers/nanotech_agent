# Human Idea Assessment App

Run with:

```powershell
streamlit run assement_app/app.py
```

You can also run it from inside `assement_app/` with:

```powershell
streamlit run app.py
```

The app expects an `assessment_bundle_v1` JSON file produced by `novelty_app.evaluation.run_retrospective`.

Reviewer state is written to an Excel workbook with four sheets:

- `meta`
- `ideas`
- `assessments`
- `summary`
