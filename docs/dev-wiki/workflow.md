# Local Workflow

## What Law/Luigi Means Here

Luigi provides the task/dependency model. Law adds HEP workflow features such as local/HTCondor workflows, branch maps, targets, and file localization.

In this repo:

- A task is a Python class with `requires`, `output`, `run`, and usually `create_branch_map`.
- A branch is one unit of parallel work, such as one dataset, one input ROOT file, or one histogram variable.
- A target is an output file. If it exists and `complete()` is true, law skips the task.
- `--version` namespaces outputs so you can run the same workflow many times.
- `--period` selects era-specific config.
- `--workflow local` runs locally. `--workflow htcondor` submits branch jobs.

The shared base class is `FLAF/run_tools/law_customizations.py::Task`.

## Setup

```sh
source env.sh
law index --verbose
voms-proxy-init -voms cms -rfc -valid 192:00
```

Create `config/user_custom.yaml` before real production. It usually defines `fs_*` paths, `analysis_config_area`, booleans for uncertainty production, and `variables`.

For local learning, keep `variables` tiny.

```yaml
analysis_config_area: config
compute_unc_variations: false
compute_unc_histograms: false
store_noncentral: false
variables:
  - mu1_pt
```

## Concrete Task Chain

Some older docs mention umbrella names like `AnaTupleTask` and `AnaCacheTask`. In this checkout, the concrete task classes are these:

| Task | Branch Meaning | Main Script | Main Output |
| --- | --- | --- | --- |
| `InputFileTask` | one dataset | none | `data/<version>/InputFileTask/<period>/<dataset>.json` |
| `AnaTupleFileTask` | one input NanoAOD file | `FLAF/AnaProd/anaTupleProducer.py` | `<version>/AnaTuples_split/<period>/<dataset>/anaTupleFile_*.root` |
| `AnaTupleFileListBuilderTask` | one dataset or merged data pseudo-dataset | `FLAF/AnaProd/AnaTupleFileList.py` | merge plans and reports |
| `AnaTupleFileListTask` | local copy of merge plan | none | local merge plan target |
| `AnaTupleMergeTask` | one merged chunk | `FLAF/AnaProd/MergeAnaTuples.py` | `<version>/AnaTuples/<period>/<dataset>/*.root` |
| `AnalysisCacheTask` | one payload producer over one hist tuple branch | `FLAF/Analysis/AnalysisCacheProducer.py` | `<version>/AnalysisCache/<producer>/<period>/<dataset>/*` |
| `AnalysisCacheAggregationTask` | one sample for one aggregate producer | `FLAF/Analysis/AnalysisCacheAggregator.py` | local aggregate cache |
| `HistTupleProducerTask` | one merged ana tuple file | `FLAF/Analysis/HistTupleProducer.py` | `<version>/HistTuples/<period>/<dataset>/*.root` |
| `HistFromNtupleProducerTask` | one variable for one dataset | `FLAF/Analysis/HistProducerFromNTuple.py` | `<version>/Hists_split/<period>/<var>/<dataset>.root` |
| `HistMergerTask` | one variable | `FLAF/Analysis/HistMergerFromHists.py` | `<version>/Hists_merged/<period>/<var>/<var>.root` |
| `HistPlotTask` | one variable | `FLAF/Analysis/HistPlotter.py` | `<version>/Plots/<period>/<var>/<region>/<category>/*.pdf` |

## Good First Local Inspection

Use `--print-status` before launching work:

```sh
law run InputFileTask --period Run3_2022 --version dev --print-status 2,0
law run HistPlotTask --period Run3_2022 --version dev --print-status 3,1
```

Narrow the problem:

```sh
law run HistPlotTask \
  --period Run3_2022 \
  --version dev \
  --dataset GluGluHto2Mu \
  --test 100 \
  --workflow local \
  --print-status 3,1
```

Useful parameters inherited by all tasks:

| Parameter | Meaning |
| --- | --- |
| `--period Run3_2022` | selects `config/Run3_2022/*` |
| `--version dev` | output namespace |
| `--dataset NAME` | keep only matching dataset names |
| `--process NAME` | keep only matching process names |
| `--model NAME` | use a different physics model from `phys_models.yaml` |
| `--customisations a.b=c;x.y=z` | override config keys at runtime |
| `--test N` | limit event/input-file work where implemented |
| `--branches 0,2-4` | run selected task branches |

## How To Read A Failed Workflow

1. Identify the failing task class in the law output.
2. Open that class in `FLAF/AnaProd/tasks.py` or `FLAF/Analysis/tasks.py`.
3. Read `create_branch_map` to understand what the branch data tuple means.
4. Read `requires` to see which upstream output is being localized.
5. Read `run` to find the producer script and exact command arguments.
6. Open the producer script. That is where the event loop is.
7. Search for the output path pattern in `output`.

Most errors fall into four buckets:

- Config selection: `Setup` did not select the process, dataset, variable, or file system you expected.
- Missing upstream target: task dependency did not run or output path points at a different `--version`.
- RDataFrame column error: a `Define` expression references a missing or wrongly typed column.
- Correction stage mismatch: correction exists in config but is not active for this stage.

## Local Output Map

Local law bookkeeping lives under:

```text
data/<version>/<TaskName>/<period>/
data/jobs/
.law/
```

Physics outputs may be local or remote depending on `fs_*` keys. If an `fs_*` value starts with `/`, law writes a local file target. Otherwise it builds a WLCG file system target.

## Where The Workflow Touches Your Analysis Code

Ana tuple production:

- `FLAF/AnaProd/anaTupleProducer.py`
- `AnaProd/anaTupleDef.py`
- `AnaProd/baseline.py`
- `Corrections/Corrections.py`

Histogram production:

- `FLAF/Analysis/HistTupleProducer.py`
- `Analysis/histTupleDef.py`
- `Analysis/H_mumu.py`
- `Analysis/MuonRelatedFunctions.py`
- `Analysis/JetRelatedFunctions.py`
- `Corrections/Corrections.py`

ML payload cache:

- `FLAF/Analysis/AnalysisCacheProducer.py`
- `config/global.yaml::payload_producers`
- `Analysis/*_Application.py`
