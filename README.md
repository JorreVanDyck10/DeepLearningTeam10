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

These are starter templates, not implemented pipelines. The AWS requirement applies to at least one model across the project; document when that notebook does not apply to a dataset. Deployment can start with the baseline and later use the selected model.

Within each dataset folder:

- `data/raw/`: original downloaded data (ignored by Git).
- `data/processed/`: prepared data (ignored by Git).
- `models/`: saved models and preprocessing (ignored by Git).
- `metrics/`: small evaluation results to commit for comparison.

Run notebooks with the dataset folder as the working directory so relative paths such as `data/raw/` resolve consistently. Add package dependencies as implementation choices are made.

## Notebook conventions

Start every notebook with the contributors and what each person did. Explain each code block in a preceding Markdown cell. Keep one notebook per individual model, and retain numeric prefixes when renaming templates. Additional EDA notebooks can use suffixes such as `02a_...` and `02b_...` before final preparation.

Citi Bike downloads, extraction, and assembly must be performed with code. Establish a statistically supported hypothesis before modelling. Keep large data files and model artifacts out of Git.

## GenAI disclosure

Codex generated the initial folder structure, notebook templates, and structure documentation. The course GenAI policy was not provided with the assignment text; review it and complete any required disclosure as the project develops.