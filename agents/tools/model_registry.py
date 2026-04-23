# agents/tools/model_registry.py
"""
Shared model state - load 1 lần, inject cho tất cả tools.
"""
_model_dense = None
_model_res = None
_class_mapping_res = None


def init_models(model_dense, model_res, class_mapping_res):
    """Inject model instances từ Orchestrator."""
    global _model_dense, _model_res, _class_mapping_res
    _model_dense = model_dense
    _model_res = model_res
    _class_mapping_res = class_mapping_res


def get_models():
    return _model_dense, _model_res, _class_mapping_res
