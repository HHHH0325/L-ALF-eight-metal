# L-ALF-eight-metal

L-ALF applies LLM-guided multi-objective active learning to Pd-Pt-Au-Cr-Ir-Ce-Cu-Nb alloy compositions. Power density is evaluated with a fixed pretrained gradient-boosting regressor (GBR); raw-material cost is normalized to pure Pd.

Composition vectors use `[Pd, Pt, Cu, Au, Ir, Ce, Nb, Cr]`.

## Install

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD/src"
```

On macOS or Linux, activate `.venv/bin/activate` and set `PYTHONPATH="$PWD/src"`.

## Configure the LLM

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL`. Set `OPENAI_BASE_URL` for an OpenAI-compatible endpoint.

```powershell
Copy-Item .env.example .env
```

## Load the fixed GBR

Obtain the pretrained model from the code and data associated with Zhang et al., *A multimodal robotic platform for multi-element electrocatalyst discovery*, Nature 647, 390-396 (2025), [doi:10.1038/s41586-025-09640-5](https://doi.org/10.1038/s41586-025-09640-5). Place the file at `data/oracle/gradient_boosting_model.joblib` and verify its SHA-256 as described in [`data/oracle/README.md`](data/oracle/README.md). The supplied algorithms load this fixed model and do not retrain it.

## Run

Run the L-ALF pipeline (up to 30 iterations; the configured performance threshold may stop it earlier):

```powershell
python -m l_alf.run_pipeline --config configs/lalf_pipeline.json
```

Run B2 with A0 and six ablations, using 30 rounds, three evaluations per round, and five repeats:

```powershell
python -m l_alf.run_benchmark --suite B2 --config configs/module_30_rounds.example.json
python -m l_alf.run_benchmark --suite B2 --config configs/module_30_rounds_w06.example.json
```

Run a traditional baseline (150 new evaluations per configuration):

```powershell
python -m l_alf.run_baseline --config configs/baseline_mobo.json --method mobo
python -m l_alf.run_baseline --config configs/baseline_dg_mobo.json --method dg_mobo
python -m l_alf.run_baseline --config configs/baseline_nsga2.json --method nsga2
python -m l_alf.run_baseline --config configs/baseline_random.json --method random
```

The L-ALF pipeline and LLM-based B2 methods require the configured API. Traditional baselines do not require an LLM API. All runs require the fixed GBR at the path above.

## Data

[`data/README.md`](data/README.md) describes the included initial compositions, B1/B2 metric summaries, fixed-Q error summary, candidate records, and Table 1.
