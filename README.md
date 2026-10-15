# Marketing Uplift Experiment Lab

A two-model treatment-effect prototype that ranks randomized experiment holdout records by estimated response uplift.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A two-model treatment-effect prototype that ranks randomized experiment holdout records by estimated response uplift. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[MineThatData/Hillstrom email-marketing challenge data](https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-Mail Analytics_Data_Mart_Challenge_2008.7z). A CSV with randomized treatment coded 0/1, a binary conversion outcome, and pre-treatment features only. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/campaign.csv --treatment treatment --outcome conversion --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

The estimator assumes randomized assignment and no post-treatment features. Treat ranking as exploratory; verify uplift in an untouched experiment before deployment.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

No code license file is included yet. Third-party datasets remain subject to their original terms.
