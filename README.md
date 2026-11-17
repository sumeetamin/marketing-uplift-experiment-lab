# Marketing Uplift Experiment Lab

An uplift-ranking prototype on a randomized email experiment. It estimates the difference in conversion probabilities between a men's merchandise email and no email for customers with different pre-treatment characteristics.

## Data and provenance

The experiment is Kevin Hillstrom's MineThatData Email Analytics Challenge (2008). This run used a public CSV mirror at [W-Tran/uplift-modelling](https://github.com/W-Tran/uplift-modelling/blob/master/data/hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv), SHA-256 `27bab8c5d3669f26ec08ebb50a0a78317542f29501156f2e2af6781fab4cd7e2`. The source file does not include a machine-readable license; credit is retained and the raw dataset is not redistributed here.

The run compares the randomized `Mens E-Mail` arm with `No E-Mail` (42,613 records). Treatment assignment is encoded as 1/0 and `conversion` is the outcome. The assignment `segment` field and post-campaign `visit` and `spend` outcomes are excluded from model features.

## Reproduce

Download the CSV mirror, create the binary treatment contrast, then run:

```powershell
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv' -OutFile hillstrom.csv
python -m pip install -r requirements.txt
python -c "import pandas as pd; d=pd.read_csv('hillstrom.csv'); d=d[d.segment.isin(['Mens E-Mail','No E-Mail'])].copy(); d['treatment']=(d.segment=='Mens E-Mail').astype(int); d.to_csv('mens_vs_control.csv',index=False)"
$hash = (Get-FileHash .\hillstrom.csv -Algorithm SHA256).Hash.ToLower()
python main.py --input mens_vs_control.csv --treatment treatment --outcome conversion --source-sha256 $hash --out outputs
```

The runner fits separate unweighted logistic outcome models on the randomized control and treatment training partitions. It scores an untouched 25% holdout and reports observed treatment-control conversion differences overall and among the top 10% ranked by estimated uplift. Local outputs, source data, and model files are ignored by Git.

## Holdout results

The stratified split used 31,959 training and 10,654 holdout rows.

| Measure | Estimate |
|---|---:|
| Overall observed conversion lift | 0.68 percentage points |
| 95% approximate interval | 0.31 to 1.04 percentage points |
| Top-decile observed lift | 1.17 percentage points |
| Top-decile 95% interval | -0.38 to 2.72 percentage points |

The top-decile interval crosses zero, so this run does **not** show reliable evidence that the uplift ranking improves targeting over the overall randomized effect. This is not an ROI estimate; the dataset has no campaign costs or margin model. Test a cost-aware policy in a new randomized campaign before using it for outreach.

Full aggregate metrics and fingerprints: [`reports/benchmark.json`](reports/benchmark.json).
