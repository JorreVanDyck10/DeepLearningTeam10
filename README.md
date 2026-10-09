# DeepLearningTeam10
Jorre Van Dyck
Milan Wouters
Raoul Buffels
Andrew Noeyens

## Project structure

Each dataset has its own folder:

- `secondary_mushroom/`: Secondary Mushroom classification.
- `nyc_citi_bike/`: NYC Citi Bike analysis and prediction.

Open notebooks with a Python 3 Jupyter kernel and run them in numeric order within each dataset folder:

| Notebook | Purpose |
| --- | --- |
| `01_scrape.ipynb` | Download, extract, and load data using code |
| `02_clean_and_explore.ipynb` | Cleaning experiments, EDA, graphs, and statistical analysis |
| `03_prepare_data.ipynb` | Final reproducible preparation without exploratory graphs |
| `04_predict_baseline.ipynb` | Quick first model |
| `05_predict_automl.ipynb` | PyCaret or another automated model comparison |
| `06_predict_model_1.ipynb` | First additional model and tuning |
| `07_predict_model_2.ipynb` | Second additional model and tuning |
| `08_predict_aws.ipynb` | AWS training and tuning for at least one dataset |
| `09_compare_models.ipynb` | Model comparison, error analysis, and selection |
| `10_deploy.ipynb` | API, frontend, hosting, and automated model updates |

The download notebooks, the Mushroom baseline (04), and Andrew's Mushroom comparison (09) are implemented; several other numbered notebooks remain starter templates. Individual solutions and AWS exports also live in contributor folders. The AWS requirement applies to at least one model across the project; document when that notebook does not apply to a dataset. Deployment can start with the baseline and later use the selected model.

Within each dataset folder:

- `data/raw/`: original downloaded data (ignored by Git).
- `data/processed/`: prepared data (ignored by Git).
- `models/`: saved models and preprocessing (ignored by Git except the small Mushroom baseline used for the first API deployment).
- `metrics/`: small evaluation results to commit for comparison.

Run notebooks with the dataset folder as the working directory so relative paths such as `data/raw/` resolve consistently. Add package dependencies as implementation choices are made.

## Notebook conventions

Start every notebook with the contributors and what each person did. Explain each code block in a preceding Markdown cell. Keep one notebook per individual model, and retain numeric prefixes when renaming templates. Additional EDA notebooks can use suffixes such as `02a_...` and `02b_...` before final preparation.

Citi Bike downloads, extraction, and assembly must be performed with code. Establish a statistically supported hypothesis before modelling. Keep large data files and model artifacts out of Git.

## GenAI disclosure

Codex generated the initial folder structure, notebook templates, structure documentation, and dataset download code. The course GenAI policy was not provided with the assignment text; review it and complete any required disclosure as the project develops.

## Download the datasets

Run `01_scrape.ipynb` in each dataset folder, or run the following from the repository root with Python 3.11 or later (no extra packages needed for the script):

```sh
python download_data.py
```

This downloads the complete UCI Secondary Mushroom dataset and all NYC Citi Bike CSV parts for January 2025, extracts the archives, and assembles each dataset locally. Citi Bike's initial archive is approximately 414 MB; allow several GB of free disk space for archives, extracted files, and the assembled CSV. One winter month is an initial scope, not a representative full year.

To select additional months:

```sh
python download_data.py --dataset nyc_citi_bike --months 202501 202502 202503
```

Outputs:

- `secondary_mushroom/data/raw/secondary_mushroom.csv` (semicolon delimiter).
- `nyc_citi_bike/data/raw/citibike_trips.csv` (comma delimiter).
- A `.manifest.json` alongside each CSV records sources, archive hashes, columns, and row counts.

Existing archives are reused. Combined CSVs are rebuilt from the selected sources, so reruns do not append duplicate rows. No cleaning or deduplication occurs at this stage. Sources: [UCI](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset) and [Citi Bike](https://citibikenyc.com/system-data). Data and manifests stay local and are ignored by Git.

Verified default downloads: Mushroom has 61,069 rows (21 columns); Citi Bike has 2,124,475 rows (13 columns). The January archive includes 207 trips starting in December 2024; these original records are retained for a documented date-filtering decision during cleaning. Both download notebooks were executed successfully, including cached reruns.


## First Mushroom prediction model

`secondary_mushroom/04_predict_baseline.ipynb` trains a small decision tree (maximum depth 5) with missing-value handling and one-hot encoding in a single pipeline. It reads the original download directly for this initial baseline; the full EDA and preparation notebooks remain to be developed. Exact duplicate rows are removed before a reproducible stratified 80/20 split.

Install the verified dependencies in your selected environment:

```sh
python -m pip install -r requirements-baseline.txt
```

Run the baseline notebook to save the model, metrics, and sample input. Then predict from the repository root:

```sh
python secondary_mushroom/predict.py --input secondary_mushroom/example_mushroom.json
```

The JSON contains all 20 original feature names and dataset codes; `null` represents missing values. The model pipeline is stored locally at `secondary_mushroom/models/baseline_decision_tree.joblib`. Metrics are in `secondary_mushroom/metrics/baseline_metrics.json`.

Initial test results: **73.36% accuracy** versus **55.39%** for the majority-class reference. Poisonous recall is only **58.82%**: 2,779 poisonous examples are misclassified as edible. This is an educational baseline on simulated data, unsuitable for deciding whether real mushrooms are safe to eat. No hyperparameter tuning was performed on the test set.

Codex implemented and executed this baseline and checked saved-model inference against notebook predictions.

## Backend starter

The FastAPI backend in `backend/main.py` exposes the existing Mushroom baseline through
`POST /predict/mushroom`, with request validation, startup model loading, readiness and CORS.
See [backend/README.md](backend/README.md) for local startup, a PowerShell prediction example
and Render configuration. `render.yaml` configures the first free Render deployment using
the small baseline artifact included in Git. Citi Bike inference, external artifact delivery
and automated model updates remain to be implemented. Codex generated this backend starter
and initial deployment configuration.

## Website and automatic deployment

The existing Render service now hosts both `frontend/` and the API at
https://deeplearningteam10-api.onrender.com/. Pushes to the linked `main` branch
trigger a new deployment. The website automatically calls the API on its own
domain; no separate static site is required. GitHub Actions checks the website,
API routes and committed model inference on pushes and pull requests.

See [docs/deployment.md](docs/deployment.md) for Render settings, local startup,
checks and the distinction between deploying code and retraining models.


## Provisional Mushroom model: Andrew

The live website and API currently use Andrew's previously exported 200-tree Random Forest
in `SolutionAndrew/MushroomDataset/models/random_forest.joblib`. Its metrics
and example are stored alongside that artifact. Andrew's latest notebook now defines
other models, including a 500-tree Random Forest. The API and website use the old export's
12 features, including the optional noise fields. Reproduced test accuracy is
78.2%; these figures describe Andrew's 5,000-record dataset and should not be
compared directly with scores from the full Secondary Mushroom dataset.
See [the export/deployment instructions](SolutionAndrew/MushroomDataset/DEPLOYMENT.md).

## Andrew's Mushroom model comparison

The [Dutch comparison report](secondary_mushroom/09_andrew_vergelijkingsrapport.md)
and [executed comparison notebook](secondary_mushroom/09_compare_models.ipynb)
reproduce Andrew's four current notebook models and audit his two saved pipelines.
The saved Logistic Regression has a different configuration and is evaluated separately.
All candidates use the same 4,100 training records, five validation folds and 900 test records.

RF500 is the provisional choice on validation F1 for the poisonous class. Its test accuracy
is 81.00%, versus 80.89% for Gradient Boosting; both miss 114 of 341 poisonous records.
These results describe Andrew's 5,000-row, 12-feature version and do not establish the final
winner against full-UCI/AWS models. The comparison does not replace the deployed RF200.
See the report for error analysis, limitations and reproduction in a separate environment.

## Citi Bike API

The published backend now also provides `POST /predict/citibike`, using the saved
January 2025 hourly decision-tree baseline. The website sends date and hour and
shows a prediction plus the 24-hour chart. The model uses only hour and weekday;
predictions outside January 2025 include a warning. See
[nyc_citi_bike/README.md](nyc_citi_bike/README.md) for evaluation and training.
