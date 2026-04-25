import os
import sys


ana_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ana_path not in sys.path:
    sys.path.insert(0, ana_path)

from FLAF.Analysis.arrow_cache import (
    ArrowCacheWriter,
    dataframe_to_awkward_via_numpy,
    read_arrow_table,
)


def _toy_tables():
    import pyarrow as pa

    yield pa.table(
        {
            "FullEventId": [101, 102, 103],
            "DNN_NNOutput": [0.10, 0.25, 0.90],
        }
    )
    yield pa.table(
        {
            "FullEventId": [104, 105],
            "DNN_NNOutput": [0.55, 0.80],
        }
    )


def run_toy_workflow(out_path):
    with ArrowCacheWriter(out_path, save_as="arrow") as writer:
        for table in _toy_tables():
            writer.write_table(table)

    table = read_arrow_table(out_path, save_as="arrow")
    assert table.num_rows == 5
    assert table.column_names == ["FullEventId", "DNN_NNOutput"]
    assert table["FullEventId"].to_pylist() == [101, 102, 103, 104, 105]
    return table


def test_arrow_cache_roundtrip(tmp_path):
    run_toy_workflow(str(tmp_path / "toy_cache.arrow"))


def test_numpy_materialization_to_awkward():
    import numpy as np

    class FakeDataFrame:
        def AsNumpy(self, columns):
            assert columns == ["FullEventId", "feature"]
            return {
                "FullEventId": np.array([11, 12]),
                "feature": np.array([1.5, 2.5]),
            }

    array = dataframe_to_awkward_via_numpy(FakeDataFrame(), ["FullEventId", "feature"])
    assert array.fields == ["FullEventId", "feature"]
    assert array["FullEventId"].to_list() == [11, 12]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "toy_cache.arrow")
        table = run_toy_workflow(out_path)
        print(f"wrote {out_path}")
        print(f"rows {table.num_rows}")
        print(f"columns {','.join(table.column_names)}")
