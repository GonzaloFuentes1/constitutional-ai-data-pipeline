import os
import random
import types

import pytest

import utils
from utils import (
    build_chat_with_fewshots,
    build_clean_chat,
    chats_to_prompts,
    chunked,
    create_output_paths,
    ensure_pad_token,
    parse_gpu_config,
    set_seed,
    setup_vllm_environment,
)


class TestChunked:
    def test_even_split(self):
        assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_last_chunk_is_smaller(self):
        assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_size_larger_than_list(self):
        assert list(chunked([1, 2, 3], 10)) == [[1, 2, 3]]

    def test_size_one(self):
        assert list(chunked(["a", "b"], 1)) == [["a"], ["b"]]

    def test_empty_list(self):
        assert list(chunked([], 4)) == []

    def test_is_lazy_generator(self):
        gen = chunked([1, 2, 3], 2)
        assert isinstance(gen, types.GeneratorType)
        assert next(gen) == [1, 2]

    def test_does_not_modify_input(self):
        data = [1, 2, 3, 4, 5]
        list(chunked(data, 2))
        assert data == [1, 2, 3, 4, 5]

    def test_zero_size_is_rejected(self):
        with pytest.raises(ValueError):
            list(chunked([1, 2, 3], 0))


class TestBuildChatWithFewshots:
    def test_no_fewshots_returns_only_user_prompt(self):
        assert build_chat_with_fewshots([], "hello") == [
            {"role": "user", "content": "hello"}
        ]

    def test_fewshots_come_first_in_order(self):
        fewshots = [
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
            [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        ]
        chat = build_chat_with_fewshots(fewshots, "final")
        assert [m["content"] for m in chat] == ["q1", "a1", "q2", "a2", "final"]
        assert [m["role"] for m in chat] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]

    def test_missing_keys_get_defaults(self):
        chat = build_chat_with_fewshots([[{}, {"content": "x"}]], "p")
        assert chat[0] == {"role": "user", "content": ""}
        assert chat[1] == {"role": "user", "content": "x"}

    def test_extra_keys_are_dropped(self):
        chat = build_chat_with_fewshots(
            [[{"role": "assistant", "content": "a", "extra": 1}]], "p"
        )
        assert chat[0] == {"role": "assistant", "content": "a"}

    def test_input_is_not_mutated_and_messages_are_copied(self):
        fewshots = [[{"role": "user", "content": "q"}]]
        chat = build_chat_with_fewshots(fewshots, "p")
        chat[0]["content"] = "changed"
        assert fewshots == [[{"role": "user", "content": "q"}]]

    def test_each_call_returns_a_new_list(self):
        first = build_chat_with_fewshots([], "p")
        second = build_chat_with_fewshots([], "p")
        first.append({"role": "assistant", "content": "x"})
        assert len(second) == 1


def test_build_clean_chat():
    assert build_clean_chat("hi") == [{"role": "user", "content": "hi"}]


class TestParseGpuConfig:
    def test_multiple_gpus(self):
        assert parse_gpu_config("0,1,2") == (3, ["0", "1", "2"])

    def test_single_gpu(self):
        assert parse_gpu_config("1") == (1, ["1"])

    def test_whitespace_and_empty_items_are_ignored(self):
        assert parse_gpu_config(" 0 , 1,, ") == (2, ["0", "1"])

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_defaults_to_gpu_zero(self, value):
        assert parse_gpu_config(value) == (1, ["0"])


class TestCreateOutputPaths:
    def test_paths_use_exps_dir_and_constitution_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils.time, "time", lambda: 1700000000.9)
        out_dir = str(tmp_path / "out")
        exps, out_json, out_parquet = create_output_paths(
            "constitutions/constitution_few.json", out_dir
        )
        assert exps == os.path.join(out_dir, "exps")
        assert os.path.isdir(exps)
        assert out_json == os.path.join(exps, "constitution_few_1700000000.json")
        assert out_parquet == os.path.join(
            exps, "constitution_few_1700000000.parquet"
        )

    def test_empty_output_dir_falls_back_to_constitution_dir(self, tmp_path):
        constitution = tmp_path / "my_constitution.json"
        exps, out_json, _ = create_output_paths(str(constitution), "")
        assert exps == os.path.join(str(tmp_path), "exps")
        assert os.path.basename(out_json).startswith("my_constitution_")


class TestSetSeed:
    def test_same_seed_gives_same_random_sequence(self):
        set_seed(123)
        first = [random.random() for _ in range(5)]
        set_seed(123)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_is_seeded_when_available(self):
        np = pytest.importorskip("numpy")
        set_seed(7)
        first = np.random.rand(3).tolist()
        set_seed(7)
        assert np.random.rand(3).tolist() == first


class FakeTokenizer:
    def __init__(self, **special):
        self.pad_token = special.get("pad_token")
        self.sep_token = special.get("sep_token")
        self.cls_token = special.get("cls_token")
        self.mask_token = special.get("mask_token")
        self.added = None

    def add_special_tokens(self, tokens):
        self.added = dict(tokens)

    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{m['role']}:{m['content']}" for m in chat) + "|assistant:"


class TestTokenizerHelpers:
    def test_ensure_pad_token_adds_only_missing_tokens(self):
        tok = FakeTokenizer(pad_token="<pad>", sep_token="<sep>")
        ensure_pad_token(tok)
        assert tok.added == {"cls_token": "", "mask_token": ""}

    def test_ensure_pad_token_adds_pad_when_missing(self):
        tok = FakeTokenizer(sep_token="s", cls_token="c", mask_token="m")
        ensure_pad_token(tok)
        assert tok.added == {"pad_token": "[PAD]"}

    def test_ensure_pad_token_noop_when_all_present(self):
        tok = FakeTokenizer(
            pad_token="p", sep_token="s", cls_token="c", mask_token="m"
        )
        ensure_pad_token(tok)
        assert tok.added is None

    def test_chats_to_prompts_applies_template_to_each_chat(self):
        chats = [
            [{"role": "user", "content": "a"}],
            [
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
        ]
        assert chats_to_prompts(FakeTokenizer(), chats) == [
            "user:a|assistant:",
            "user:b|assistant:c|assistant:",
        ]

    def test_chats_to_prompts_empty(self):
        assert chats_to_prompts(FakeTokenizer(), []) == []


def test_setup_vllm_environment_sets_variables(monkeypatch):
    # Register the variables with monkeypatch first so they are restored afterwards.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    setup_vllm_environment("2,3")
    assert os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_setup_vllm_environment_without_gpus_leaves_cuda_untouched(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    setup_vllm_environment("")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
