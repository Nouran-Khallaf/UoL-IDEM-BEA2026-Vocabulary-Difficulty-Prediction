class PipelineError(Exception):
    """
    Base exception for all pipeline-related errors.
    """

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} | context: {context_str}"


class ConfigError(PipelineError):
    """
    Raised when configuration is missing, malformed, inconsistent,
    or fails validation.
    """


class DataValidationError(PipelineError):
    """
    Raised when raw or processed data fails schema/content validation.
    """


class FeatureValidationError(PipelineError):
    """
    Raised when feature generation or feature integrity checks fail.
    """


class ModelConfigurationError(PipelineError):
    """
    Raised when a model config is invalid or incompatible with the experiment.
    """


class ExperimentRuntimeError(PipelineError):
    """
    Raised when an experiment fails during execution.
    """


class MetricComputationError(PipelineError):
    """
    Raised when metric computation fails or receives invalid inputs.
    """