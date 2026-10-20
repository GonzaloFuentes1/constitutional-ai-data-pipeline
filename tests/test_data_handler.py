import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from data_handler import (
    load_base_prompt,
    load_constitution,
    load_red_teaming_prompts_from_parquet,
    save_constitution_with_results,
    save_parquet_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
CONSTITUTION_FILES = sorted((ROOT / "constitutions").glob("*.json"))


def valid_constitution():
    return {
        "constitutions": [
            {"critic": "Is it harmful?", "revision": "Rewrite it safely."}
        ],
        "few_shot_examples": [
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        ],
    }


def write_json(tmp_path, data, name="constitution.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestLoadConstitution:
    def test_valid_file_is_returned_unchanged(self, tmp_path):
        data = valid_constitution()
        assert load_constitution(write_json(tmp_path, data)) == data

    def test_few_shot_examples_default_to_empty_list(self, tmp_path):
        data = valid_constitution()
        del data["few_shot_examples"]
        loaded = load_constitution(write_json(tmp_path, data))
        assert loaded["few_shot_examples"] == []

    def test_empty_few_shot_list_is_accepted(self, tmp_path):
        data = valid_constitution()
        data["few_shot_examples"] = []
        assert load_constitution(write_json(tmp_path, data))["few_shot_examples"] == []

    def test_all_roles_are_accepted(self, tmp_path):
        data = valid_constitution()
        data["few_shot_examples"] = [
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        ]
        assert load_constitution(write_json(tmp_path, data)) == data

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda d: d.pop("constitutions"), "constitutions"),
            (lambda d: d.update(constitutions={"critic": "x"}), "constitutions"),
            (lambda d: d.update(constitutions=[]), "vac"),
            (lambda d: d.update(constitutions=["not an object"]), "objeto"),
            (
                lambda d: d.update(constitutions=[{"critic": "only critic"}]),
                "critic",
            ),
            (
                lambda d: d.update(constitutions=[{"revision": "only revision"}]),
                "critic",
            ),
            (
                lambda d: d.update(constitutions=[{"critic": "  ", "revision": "r"}]),
                "critic",
            ),
            (
                lambda d: d.update(constitutions=[{"critic": "c", "revision": ""}]),
                "revision",
            ),
            (
                lambda d: d.update(constitutions=[{"critic": 1, "revision": "r"}]),
                "critic",
            ),
            (lambda d: d.update(few_shot_examples="nope"), "few_shot_examples"),
            (lambda d: d.update(few_shot_examples=["nope"]), "few_shot_examples"),
            (lambda d: d.update(few_shot_examples=[["nope"]]), "few_shot_examples"),
            (
                lambda d: d.update(few_shot_examples=[[{"role": "user"}]]),
                "role",
            ),
            (
                lambda d: d.update(
                    few_shot_examples=[[{"role": "tool", "content": "x"}]]
                ),
                "inv",
            ),
            (
                lambda d: d.update(
                    few_shot_examples=[[{"role": "user", "content": 3}]]
                ),
                "content",
            ),
        ],
    )
    def test_invalid_structure_raises_value_error(self, tmp_path, mutate, message):
        data = valid_constitution()
        mutate(data)
        with pytest.raises(ValueError, match=message):
            load_constitution(write_json(tmp_path, data))

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_constitution(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_constitution(str(tmp_path / "missing.json"))

    @pytest.mark.parametrize("path", CONSTITUTION_FILES, ids=lambda p: p.name)
    def test_bundled_constitutions_are_valid(self, path):
        data = load_constitution(str(path))
        assert data["constitutions"]
        assert isinstance(data["few_shot_examples"], list)


def test_save_constitution_with_results_roundtrip(tmp_path):
    data = valid_constitution()
    original = copy.deepcopy(data)
    chats = [[{"role": "user", "content": "héllo"}]]
    out = tmp_path / "out.json"

    save_constitution_with_results(data, chats, str(out))

    assert data == original  # input is not mutated
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["system_chat"] == chats
    assert saved["constitutions"] == original["constitutions"]
    assert "héllo" in out.read_text(encoding="utf-8")  # ensure_ascii=False


def test_load_base_prompt_strips_whitespace(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("\n  base prompt  \n", encoding="utf-8")
    assert load_base_prompt(str(path)) == "base prompt"


class TestParquet:
    @pytest.fixture(autouse=True)
    def _require_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_load_prompts_filters_null_and_blank_values(self, tmp_path):
        import pandas as pd

        path = tmp_path / "prompts.parquet"
        pd.DataFrame({"text": ["  first  ", None, "", "   ", "second"]}).to_parquet(
            path
        )
        assert load_red_teaming_prompts_from_parquet(str(path)) == ["first", "second"]

    def test_load_prompts_requires_text_column(self, tmp_path):
        import pandas as pd

        path = tmp_path / "prompts.parquet"
        pd.DataFrame({"prompt": ["x"]}).to_parquet(path)
        with pytest.raises(ValueError, match="text"):
            load_red_teaming_prompts_from_parquet(str(path))

    def test_load_prompts_wraps_read_errors(self, tmp_path):
        with pytest.raises(ValueError, match="Error cargando"):
            load_red_teaming_prompts_from_parquet(str(tmp_path / "missing.parquet"))

    def test_save_parquet_dataset_columns_and_messages(self, tmp_path):
        import pandas as pd

        out = tmp_path / "dataset.parquet"
        save_parquet_dataset(
            red_prompts=["p1", "p2"],
            init_responses=["i1", "i2"],
            critic_prompts=["cp1", "cp2"],
            critic_responses=["cr1", "cr2"],
            revision_prompts=["rp1", "rp2"],
            revision_responses=["rr1", "rr2"],
            output_path=str(out),
        )
        df = pd.read_parquet(out)

        assert list(df.columns) == [
            "init_prompt",
            "init_response",
            "critic_prompt",
            "critic_response",
            "revision_prompt",
            "revision_response",
            "prompt",
            "messages",
            "chosen",
            "rejected",
        ]
        assert len(df) == 2
        row = df.iloc[0]
        assert row["prompt"] == "p1"
        assert row["chosen"] == "rr1"
        assert row["rejected"] == "i1"
        messages = [dict(m) for m in row["messages"]]
        assert messages == [
            {"role": "user", "content": "p1"},
            {"role": "assistant", "content": "i1"},
            {"role": "user", "content": "cp1"},
            {"role": "assistant", "content": "cr1"},
            {"role": "user", "content": "rp1"},
            {"role": "assistant", "content": "rr1"},
        ]
