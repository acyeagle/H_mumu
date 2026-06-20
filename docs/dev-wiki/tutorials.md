# Tutorials

## Tutorial 1: Trace `mu1_pt`

Goal: follow one familiar scalar from config to plot.

1. Put it in the variable list.

   ```yaml
   variables:
     - mu1_pt
   ```

   In CI this appears in `config/ci_custom.yaml`. In local work it usually lives in `config/user_custom.yaml`.

2. Find where it is born.

   `AnaProd/anaTupleDef.py::addAllVariables` loops over the two muon legs and defines leg variables. For `pt`, it writes:

   ```text
   mu1_pt = Muon_p4[mu1_idx].pt()
   ```

   `mu1_idx` comes from `AnaProd/baseline.py::LeptonsSelection`.

3. Find when it is saved.

   `addAllVariables` uses `DefineAndAppend`, so `mu1_pt` enters `dfw.colToSave`. `FLAF/AnaProd/anaTupleProducer.py` snapshots those columns into ana tuple ROOT files.

4. Find how it becomes a histogram input.

   `FLAF/Analysis/HistTupleProducer.py` reads `global_params["variables"]`, finds binning for `mu1_pt` in `config/plot/histograms.yaml`, and defines `mu1_pt_bin`.

5. Find how it becomes a histogram and plot.

   `HistFromNtupleProducerTask` creates split hist files for `mu1_pt`, `HistMergerTask` merges them, and `HistPlotTask` makes PDFs for each channel/category/region.

## Tutorial 2: Trace `muScaRe`

Goal: understand a shape correction.

1. Config begins in `config/global.yaml`.

   ```yaml
   corrections:
     muScaRe:
       stage: AnaTuple
       mu_pt_for_ScaReApplication: "bsConstrainedPt"
       apply_scare: true
       apply_fsr_recovery: true
   ```

2. `FLAF/AnaProd/anaTupleProducer.py` initializes:

   ```text
   Corrections.initializeGlobal(stage="AnaTuple", ...)
   ```

3. `Corrections.__init__` includes `muScaRe` because its stage matches `AnaTuple`.

4. `Corrections.applyScaleUncertainties` sees `muScaRe` and calls:

   ```text
   self.muScaRe.getP4Variations(df, source_dict)
   ```

5. The provider lives in `Corrections/MuonEnergyScale_corr.py`.

6. Later, `Baseline.SelectRecoP4` chooses the systematic-specific p4 columns for this loop iteration.

7. If noncentral outputs are enabled, `anaTupleProducer.py` snapshots trees like:

   ```text
   Events__ScaRe__Up
   Events__ScaRe__Down
   ```

## Tutorial 3: Trace A Plot Category

Goal: understand category definitions.

1. Category names live in `config/global.yaml::categories`.
2. Category formulas live in `config/global.yaml::category_definition`.
3. `Analysis/H_mumu.py::DataFrameBuilderForHistograms.defineCategories` formats formulas with values like:
   - `singleMu_th[period]`
   - `WP_to_use`
   - `mu_pt_for_selection`
4. `Analysis/H_mumu.py::createKeyFilterDict` combines:
   - channel, e.g. `muMu`
   - trigger, e.g. `HLT_singleMu`
   - region, e.g. `Signal_Fit`
   - category, e.g. `VBF_JetVeto`
5. Plot tasks use these keys to produce one output per channel/category/region.

## Tutorial 4: Add A Small Workflow Feature

Good first workflow changes are usually in task plumbing, not physics math.

Example: add a new payload producer.

1. Add a producer class in `Analysis/<Name>_Application.py`.
2. Register it in `config/global.yaml::payload_producers`.
3. Pick:
   - `columns`
   - `dependencies`
   - `awkward_based`
   - `save_as`
   - `n_cpus`
   - `max_runtime`
4. Add a variable like `<Producer>_<column>` to `variables`.
5. `Setup.var_producer_map` will map that variable to the producer.
6. `HistTupleProducerTask` will require `AnalysisCacheTask` for that producer.
7. The cache file will be passed to `HistTupleProducer.py` as a friend tree.

## Tutorial 5: Add A Correction Safely

1. Start with the stage.

   Ask: should this run while making ana tuples, merging ana tuples, making analysis caches, or making hist tuples?

2. Add the config at the narrowest correct level.

   - all samples: `config/global.yaml`
   - one process: `config/<period>/processes.yaml`
   - one dataset: `config/<period>/datasets.yaml`

3. Confirm it appears in logs:

   ```text
   Corrections to apply: ...
   ```

4. Add the provider or call site.

   - p4/MET variations: `Corrections.applyScaleUncertainties`
   - normalisation weights: `Corrections.getNormalisationCorrections`
   - saved analysis columns: `AnaProd/anaTupleDef.py`
   - histogram/category effects: `Analysis/H_mumu.py`

5. Run a tiny local check with one dataset, one variable, and `--test`.

## Questions To Ask Before Changing Code

- Which stage owns this behavior?
- Is this a branch-map/dependency problem or an event-column problem?
- Is the column needed only transiently, or must it be saved?
- Does this belong in config rather than Python?
- Does data behave differently from MC?
- Does the correction need central only, Up/Down, or relative weights?
- Does a process-level processor already solve this, especially for stitching or denominators?
