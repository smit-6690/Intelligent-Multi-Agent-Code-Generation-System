import json

from magent_codegen.config import Settings
from magent_codegen.evaluation import evaluate_tasks, load_tasks
from magent_codegen.pipeline import MultiAgentPipeline


def test_evaluation_names_smoke_metric(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([{"task_id": "one", "prompt": "Return the input."}]))
    tasks = load_tasks(tasks_path)
    report = evaluate_tasks(
        MultiAgentPipeline(Settings(model_backend="mock")), tasks
    )
    assert report["metric_name"] == "pipeline_smoke_pass_rate"
    assert report["evaluated"] == 1
