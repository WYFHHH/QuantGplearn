__version__ = '1.0.0'

__all__ = ['genetic', 'functions', 'fitness']

try:
    from .gpu_transformer import GpuSymbolicTransformer
except Exception:
    GpuSymbolicTransformer = None

__all__ = ['genetic', 'functions', 'fitness', 'GpuSymbolicTransformer']
