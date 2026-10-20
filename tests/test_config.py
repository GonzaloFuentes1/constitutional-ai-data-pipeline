import pytest

from config import Args, validate_args


@pytest.fixture
def constitution_file(tmp_path):
    path = tmp_path / "constitution.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


def make_args(constitution_file, **overrides):
    return Args(constitution_path=constitution_file, **overrides)


def test_defaults_match_documented_values():
    args = Args()
    assert args.gpus == "0"
    assert args.output_dir == "outputs"
    assert args.max_samples == 128
    assert args.batch_size == 32
    assert args.tensor_parallel_size == 0
    assert len(args.red_prompts) == 20


def test_red_prompts_default_is_not_shared_between_instances():
    first, second = Args(), Args()
    first.red_prompts.append("extra")
    assert "extra" not in second.red_prompts


def test_valid_args_pass(constitution_file):
    validate_args(make_args(constitution_file))


def test_all_samples_flag_is_accepted(constitution_file):
    validate_args(make_args(constitution_file, max_samples=-1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model": "   "},
        {"max_samples": 0},
        {"max_samples": -2},
        {"batch_size": 0},
        {"max_new_tokens": 0},
        {"max_model_len": 0},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"gpu_memory_utilization": 0.0},
        {"gpu_memory_utilization": 1.1},
        {"tensor_parallel_size": -1},
        {"gpus": ","},
        {"gpus": "0,a"},
        {"gpus": "0;1"},
    ],
)
def test_invalid_values_raise_value_error(constitution_file, overrides):
    with pytest.raises(ValueError):
        validate_args(make_args(constitution_file, **overrides))


@pytest.mark.parametrize(
    "overrides", [{"gpus": "0,1,2,3"}, {"gpus": ""}, {"temperature": 0.0}, {"top_k": 0}]
)
def test_boundary_values_are_accepted(constitution_file, overrides):
    validate_args(make_args(constitution_file, **overrides))


def test_missing_constitution_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_args(Args(constitution_path=str(tmp_path / "missing.json")))


def test_missing_red_teaming_parquet_raises(constitution_file, tmp_path):
    args = make_args(
        constitution_file, red_teaming_parquet=str(tmp_path / "missing.parquet")
    )
    with pytest.raises(FileNotFoundError):
        validate_args(args)
