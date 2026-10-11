from magent_codegen.config import Settings
from magent_codegen.pipeline import MultiAgentPipeline


def test_mock_pipeline_generates_valid_code():
    pipeline = MultiAgentPipeline(Settings(model_backend="mock"))
    result = pipeline.run("Return all prime numbers up to n.")
    assert result.passed
    assert "def solve" in result.code
    assert result.repair_attempts == 0


def test_mock_pipeline_is_deterministic():
    settings = Settings(model_backend="mock")
    first = MultiAgentPipeline(settings).run("Return the input value.")
    second = MultiAgentPipeline(settings).run("Return the input value.")
    assert first.code == second.code
