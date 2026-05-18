"""Generic PyTorch-only fallback backend (no GPU dependency)."""
from .backend import GenericBackend

__all__ = ["GenericBackend"]
