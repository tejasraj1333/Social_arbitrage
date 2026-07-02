"""Domain error hierarchy.

Feature modules raise these instead of bare Exceptions so the API layer can
map them to HTTP status codes in one place (see api/app.py exception handlers).
"""

from __future__ import annotations


class SAMError(Exception):
    """Base class for all application errors."""


class ConfigError(SAMError):
    """Invalid or missing configuration."""


class CredentialsMissing(ConfigError):
    """A source's API credentials are absent; the collector cannot run live."""


class IngestionError(SAMError):
    """A collector failed to fetch or persist data."""


class NotFoundError(SAMError):
    """A requested resource (entity, signal, report) does not exist."""


class ValidationError(SAMError):
    """Input failed domain validation."""
