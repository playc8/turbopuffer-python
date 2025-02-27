import json
import os
import time
from typing import Optional

# Read API base url from env or default
api_base_url = os.environ.get("TURBOPUFFER_API_BASE_URL", "https://api.turbopuffer.com")
# Read API key from env
api_key = os.environ.get("TURBOPUFFER_API_KEY")
# Read connection timeout from env (in seconds) or default to 5
connect_timeout = float(os.environ.get("TURBOPUFFER_CONNECT_TIMEOUT", "5"))
# Read read timeout from env (in seconds) or default to 60
read_timeout = float(os.environ.get("TURBOPUFFER_READ_TIMEOUT", "60"))
# Read how many upsert calls we're willing to retry
max_retries = int(os.environ.get("TURBOPUFFER_MAX_RETRIES", "6"))
# Batch size for upsert calls
upsert_batch_size = int(os.environ.get("TURBOPUFFER_UPSERT_BATCH_SIZE", "1000"))

from .error import APIError, AuthenticationError, raise_api_error

# Expose async namespace interface
# Expose core classes
from .namespace import (
    AttributeSchema,
    CmekDict,
    EncryptionDict,
    FullTextSearchParams,
    Namespace,
    NamespaceSchema,
    anamespaces,
    namespaces,
)
from .query import (
    ConsistencyDict,
    ConsistencyLevel,
    Filter,
    Filters,
    Op,
    RankInput,
    VectorQuery,
)
from .vectors import Cursor, VectorColumns, VectorResult, VectorRow

# Read version from version.py
from .version import VERSION


def dump_json_bytes(value) -> bytes:
    """
    Internal helper for serializing json in a consistent way.
    """
    start = time.monotonic()
    json_bytes = json.dumps(
        value,
        # Only use features in python stdlib:
        ensure_ascii=False,  # We want utf-8 output
        check_circular=False,  # We know we don't have circular references
        allow_nan=True,  # Safely encode special floating values
        separators=(",", ":"),  # No spaces after separators
        sort_keys=True,  # Predictable key ordering
    ).encode("utf-8")
    dur = time.monotonic() - start
    if dur > 0.5:
        import warnings

        warnings.warn(
            f"turbopuffer serialization took {dur:0.3f}s for {len(json_bytes)} bytes ({len(json_bytes) / dur / 1024 / 1024:0.1f} MB/s)"
        )
    return json_bytes
