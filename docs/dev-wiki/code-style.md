# Code Style And Conventions

## General Style

This repo favors compact Python that builds ROOT `RDataFrame` expression graphs. Many physics expressions are strings because ROOT JIT-compiles them.

Expect this pattern:

```python
df = df.Define("new_col", "C++ expression")
df = df.Filter("new_col > 0", "filter name")
```

For analysis code, prefer small functions that accept and return `df` or `dfw`.

## DataFrameWrapper Pattern

`FLAF/Common/Utilities.py::DataFrameWrapper` wraps two things:

- `dfw.df`: current `RDataFrame`
- `dfw.colToSave`: list of columns to snapshot

Use:

- `Define`: define a column, do not save it.
- `DefineAndAppend`: define a column and save it.
- `RedefineAndAppend`: replace a column and save it.
- `Apply(func, ...)`: call helper functions that return a new dataframe.

If a downstream file needs a column, make sure it reaches `colToSave`.

## Naming

| Thing | Convention |
| --- | --- |
| Law tasks | `CamelCaseTask` |
| RDataFrame columns | mostly `snake_case`, sometimes framework names like `Muon_p4` |
| Muon legs | `mu1_*`, `mu2_*` |
| Systematics | `Central`, `<Source>Up`, `<Source>Down` |
| Weights | `weight_*` |
| Payload producer outputs | `<Producer>_<column>`, e.g. `DNN_NNOutput` |
| Period configs | `config/Run3_2022/global.yaml`, `weights.yaml`, `datasets.yaml`, `processes.yaml`, `triggers.yaml` |

## Config Conventions

Common config split:

- `config/global.yaml`: analysis-wide defaults.
- `config/phys_models.yaml`: model composition.
- `config/processes.yaml`: shared anchors and process snippets.
- `config/<period>/processes.yaml`: process definitions for an era.
- `config/<period>/datasets.yaml`: dataset file sources and cross section labels.
- `config/<period>/weights.yaml`: norm and shape uncertainties.
- `config/<period>/triggers.yaml`: trigger definitions.
- `config/user_custom.yaml`: local paths and run-local settings.

YAML keys beginning with `.` are treated as special helper entries by `Setup.Config` and removed from the final config dict. This is used for anchors such as processor templates.

## Python/C++ Boundary

Use Python for orchestration and column graph construction. Use C++ helpers when:

- expression logic is reused in many RDataFrame columns;
- a vector operation becomes unreadable as a string;
- ROOT needs a concrete type or enum;
- performance matters inside the event loop.

Declare C++ headers with `DeclareHeader(...)` or ROOT interpreter calls before using helper names in expressions.

## Workflow Code Conventions

In task classes:

- `create_branch_map` defines parallel units.
- `workflow_requires` defines broad branch-level prerequisites.
- `requires` defines exact branch dependencies.
- `output` defines target paths.
- `run` localizes inputs, calls producer scripts, and moves outputs into targets.

Do not put event-level physics logic in task classes. Put it in producer scripts or analysis modules.

## Comment Strategy

Since the repo keeps inline comments sparse, durable explanation should live in this wiki or in focused docs near the workflow. In hot analysis code, prefer clear function boundaries, column names, and config names.

When a comment is needed, make it explain why a surprising choice exists, not what the next line does.

## Formatting

C++ formatting is enforced through `.clang-format`.

```sh
clang-format -i include/*.h include/*.cc
clang-format --dry-run -Werror include/*.h include/*.cc
```

Python has no local formatter config in this checkout. Follow existing style: 4 spaces, direct helper functions, and minimal unrelated churn.
