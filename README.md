# Hmumu
Repository for the Hmumu analysis

## Developer wiki

New contributors can start with the [developer wiki](docs/dev-wiki/README.md) for architecture, local workflow, entry points, data flow, corrections, and tutorials.


## General Information
[Central NANOAOD production](https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv14/2024Prompt/doc_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAOD-140X_mcRun3_2024_realistic_v26-v2.html#Muon), taken from [Nano AOD documentation](https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc)

## How to install the FW
1. Setup ssh keys:
    On GitHub [settings/keys](https://github.com/settings/keys)
    On CERN GitLab [profile/keys](https://gitlab.cern.ch/-/user_settings/profile)
2. git clone --recursive git@github.com:cms-flaf/H_mumu.git
3. (optional but recommended) create a fork from the central repository:
    3.1. Go on [central repository](https://github.com/cms-flaf/H_mumu)
    3.2. Follow the instructions from [github documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for creating a fork
    3.3. In your work-area (or where you cloned the repository - workarea recommended) in the ```H_mumu``` cloned folder, follow [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork)
4. Create a file in the folder: H_mumu/config/user_custom.yaml
    The content of the file should be: - currently it's fine but it will be changed
    ```
    fs_default:
    - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/'
    fs_anaCache:
        - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/'
    fs_anaTuple:
        - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/'
    fs_anaTuple2p5:
        - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/'
    fs_nnCacheTuple:
        - 'T3_CH_CERNBOX:/store/user/prsolank/H_mumu/'
    fs_anaCacheTuple:
        - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/'
    fs_histograms:
        - 'T3_CH_CERNBOX:/store/user/vdamante/H_mumu/histograms/'

    analysis_config_area: config

    compute_unc_variations: False
    store_noncentral: False
    compute_unc_histograms: False

    variables:
        - mu1_pt
    ```

### Setup steps
1. Load environment: ```source env.sh```

2. Add law tasks to the index: ```law index```
(if you want to know more about what is happening, you can add ```--verbose```)

3. Initialize voms-proxy
```voms-proxy-init -voms cms -rfc -valid 192:00```
(if you do not have grid permission and certificate, see [this twiki](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid#BasicGrid))

### Get input files
This step consists in saving filenames in local (```data/InputFileTask/...```) and is required for all subsequent steps:
```law run InputFileTask --period ${ERA} --version ${VERSION_NAME}```
where you can either set the ${ERA} or just manually put the era name (Run3_2022,Run3_2022EE etc)
and version consists in a general name (e.g. Run3_2022)

### Get AnaCache
The anaCaches are yaml file storing inputs from both saved and non-saved events from original NanoAOD (needed for pu uncertainty)
```
law run AnaCacheTask --period ${ERA} --version ${VERSION_NAME}
```
The ${ERA} is defined as before, whereas version can differ from InputFileTask (e.g. AnaCacheVersion is  Run3_2022_v1 and InputFileTaskVersion is Run3_2022). The important is that **if they differ** you need also to add
```
 --InputFileTask-version ${INPUTFILETASK_VERSION_NAME}
```
which in this example is Run3_2022.


#### ToDo list
- redesign jet selection and store
- modify H(mm) candate selection
- check what is done for MET in Run2 analysis
- check the b-tagging WP based SF for jets
- include FSR
- add all corrections
- data/MC control plots (short term)
- setup workspace for ggH and VBF (for the moment both template fit)
- improve documentations + add github project to correctly keep trace of changes

#### R&D
- efficiency gain using other triggers (non priority)
- develop classifier optimized for Run3 (first focus on VBF, then possible extension to all categories)
    - explore different architechtures
    - validate
    - hyperparameter optimization
    - suitable to compare results w/ run2 and for run2+run3 combination
- switch to Run3 muon ID
- investigate for new production setup  (e.g. flashsim)
