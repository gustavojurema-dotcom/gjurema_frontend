import pandas as pd
import pytest

from gjurema import artifacts_io


def test_write_parquet_keeps_previous_file_on_failure(tmp_path):
    path = tmp_path / "tabela.parquet"
    artifacts_io.write_parquet(pd.DataFrame({"a": [1]}), path)

    def fail(_temp):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        artifacts_io.write_with(path, fail)

    assert pd.read_parquet(path)["a"].tolist() == [1]
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_concurrent_writes_use_distinct_temp_files(tmp_path):
    path = tmp_path / "meta.json"
    seen = []

    def nested(_temp):
        seen.append(_temp)
        if len(seen) == 1:
            artifacts_io.write_with(path, nested)
        _temp.write_text('{"gen": 1}')

    artifacts_io.write_with(path, nested)

    assert seen[0] != seen[1]
    assert path.read_text() == '{"gen": 1}'
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_write_json_publishes_atomically(tmp_path):
    path = tmp_path / "meta.json"
    artifacts_io.write_json({"ok": True}, path)

    assert path.read_text().strip().startswith("{")
    assert list(tmp_path.glob(".*.tmp*")) == []
