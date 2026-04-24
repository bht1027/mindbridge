from dataclasses import dataclass


PIPELINE_MODES = ("full", "no_critic", "no_reviser", "no_safety", "no_retrieval")


@dataclass(frozen=True)
class PipelineRunConfig:
    name: str
    use_safety: bool = True
    use_critic: bool = True
    use_reviser: bool = True
    use_retrieval: bool = True


def get_pipeline_run_config(name: str) -> PipelineRunConfig:
    configs = {
        "full": PipelineRunConfig("full"),
        "no_critic": PipelineRunConfig("no_critic", use_critic=False),
        "no_reviser": PipelineRunConfig("no_reviser", use_reviser=False),
        "no_safety": PipelineRunConfig("no_safety", use_safety=False),
        "no_retrieval": PipelineRunConfig("no_retrieval", use_retrieval=False),
    }
    try:
        return configs[name]
    except KeyError as exc:
        supported = ", ".join(PIPELINE_MODES)
        raise ValueError(f"Unsupported pipeline mode: {name}. Use one of: {supported}.") from exc
