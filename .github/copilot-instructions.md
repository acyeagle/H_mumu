# H_mumu Analysis Framework - Copilot Instructions

## Repository Overview
H→μμ (Higgs to dimuon) analysis for CMS Run 3 using FLAF (Framework for Large-scale Analysis Framework). Analyzes NanoAOD data for Higgs boson decays to muon pairs (ggH and VBF modes).

**Type**: Physics analysis | **Size**: ~3.4MB | **Languages**: Python, C++ (ROOT/RDataFrame)  
**Key Dependencies**: ROOT, law (Luigi Analysis Workflow), FLAF, Corrections frameworks  
**Channel**: muMu (dimuon)

## Project Structure

### Root Files
- `env.sh` - Setup script (sources FLAF/env.sh, sets ANALYSIS_PATH)
- `.clang-format` - C++ formatting (Google style, 120 cols, 4 spaces)
- `.editorconfig` - Editor config (4 spaces, UTF-8)

### Key Directories
- `AnaProd/` - Tuple production: `anaTupleDef.py` (observables), `baseline.py` (selection)
- `Analysis/` - Main code: `H_mumu.py` (core logic), `histTupleDef.py`, `DNN_Application.py`, `GetTriggerWeights.py`, `models/` (ONNX files)
- `include/` - C++ headers: `Helper.h` (VBF jets), `HmumuCore.h` (structures), `MuonScaRe.cc`
- `config/` - YAML configs:
  - `global.yaml` - Main config (channels, categories, corrections, regions)
  - `law.cfg` - Law workflow config
  - `signal_samples.yaml`, `phys_models.yaml`, `background_samples.yaml`
  - `Run3_2022/`, `Run3_2022EE/`, `Run3_2023/`, `Run3_2023BPix/` - Period configs (samples, processes, triggers, weights)
  - `ci_custom.yaml` - CI configuration
- `run_tools/law_customizations.py` - Custom Task class
- `Studies/DNN/` - Neural network training (configs, scripts)
- `FLAF/`, `Corrections/` - Git submodules (required)

## Git Submodules

**CRITICAL**: Initialize submodules before any work:
```bash
git submodule update --init --recursive
```
Required: `FLAF` (core framework), `Corrections` (corrections library). Needs GitHub/GitLab SSH keys.

## Environment Setup

### Prerequisites
- CMSSW environment (ROOT, CMSSW tools), Grid certificate, VOMS proxy
- Python 3: law, luigi, ROOT, awkward, onnxruntime, numpy, pyyaml, tomllib, psutil
- Storage: CERNBox (T3_CH_CERNBOX) or local

### Setup Procedure
1. **Initialize submodules**: `git submodule update --init --recursive`
2. **Source environment**: `source env.sh` (sets ANALYSIS_PATH, loads FLAF/env.sh - **always run first**)
3. **Index law tasks**: `law index` (run after setup or task changes)
4. **VOMS proxy**: `voms-proxy-init -voms cms -rfc -valid 192:00`
5. **Create** `config/user_custom.yaml` (see README.md for template with storage paths)

## Analysis Workflow

### Typical Analysis Steps

1. **Get input files**: `law run InputFileTask --period Run3_2022 --version <ver>` (creates `data/InputFileTask/`, **required first**)
2. **Generate AnaCache**: `law run AnaCacheTask --period Run3_2022 --version <ver>` (add `--InputFileTask-version <ver>` if differs)
3. **Analysis tuples**: `law run AnaTupleTask --period Run3_2022 --version <ver>`
4. **Histograms**: `law run HistPlotTask --period Run3_2022 --version <ver>`

**Law Parameters**: `--period` (Run3_2022/EE/2023/BPix), `--version`, `--customisations`, `--test`

## Code Formatting and Style

### C++ Code (Google style, 120 cols, 4 spaces)
- **Check**: `clang-format --dry-run -Werror include/*.h include/*.cc` (**run before commit**)
- **Fix**: `clang-format -i include/*.h include/*.cc`

### Python Code
4-space indentation, UTF-8, trim trailing whitespace

## GitHub Workflows (CI/CD)

**PR Checks** (must pass):
1. **Formatting Check** - C++ clang-format validation (FLAF workflow)
2. **Sanity Checks** - Repository structure validation (FLAF workflow)
3. **Integration Tests** - Trigger with `@cms-flaf-bot test` (authorized users only: kandrosov, valeriadamante, acyeagle)

## Known Issues and Workarounds

1. **Formatting violations**: Some files may have spacing issues around `==` or `+` operators. Always run `clang-format -i` before commit.
2. **Submodule failures**: Requires SSH keys for GitHub/GitLab (check settings/keys on both platforms).
3. **Missing dependencies**: law, luigi, ROOT provided by FLAF/env.sh - always `source env.sh` first.

## Analysis-Specific Details

**Selection**: muMu channel, opposite-sign muons, trigger matching (pT > 26 GeV), isolation, impact parameter cuts  
**Categories**: no_cuts, OS_sel, trigger_sel, baseline, VBF, ggH, VBF_JetVeto  
**Regions**: Z_sideband, Signal_Fit, H_sideband, Signal_ext, mass_inclusive  
**Corrections**: JEC, JER, jet horns fix (Run3: veto 2.5<|η|<3.0, pT<50), trgSF, mu, Vpt, pu, muScaRe  
**VBF Jets**: m_jj > 400 GeV, Δη > 2.5 (highest mass pair if multiple)  
**DNN**: 4 k-fold ONNX models in `Analysis/models/`, VBF vs ggH classification

## Naming Conventions
Config: `*.yaml`, Python: lowercase_underscore, Law tasks: CamelCaseTask, C++: CamelCase.h, Models: trained_model_N.onnx

## Testing and Validation

**Before Commit**:
1. Format C++: `clang-format -i include/*.h include/*.cc`
2. Verify: `clang-format --dry-run -Werror include/*.h include/*.cc`
3. Test setup: `source env.sh && law index --verbose`

**CI Requirements**: C++ formatting, repo structure checks, integration tests (if triggered)

## Important Notes

1. **Always source env.sh** before running any law commands
2. **Run law index** after environment setup or task changes
3. **Initialize submodules** before first use
4. **Format C++ code** before committing (CI will reject improperly formatted code)
5. **Create user_custom.yaml** with your storage paths before running analysis tasks
6. **Period-specific configs** must match your data: Run3_2022, Run3_2022EE, Run3_2023, Run3_2023BPix
7. **Version consistency**: If InputFileTask and AnaCacheTask versions differ, specify both

## Trust These Instructions

These instructions are comprehensive and tested. Only search for additional information if:
- Instructions are incomplete for your specific task
- You encounter errors not covered here
- Instructions appear outdated or incorrect based on error messages

For most tasks, following these instructions exactly will prevent common issues and CI failures.
