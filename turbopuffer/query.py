import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from typing_extensions import Literal

# Refer to turbopuffer docs for valid operator names
FilterOperator = str

FilterValue = Union[str, int, List[str], List[int]]
FilterCondition = Tuple[str, FilterOperator, FilterValue]

LegacyFilterCondition = Tuple[FilterOperator, FilterValue]
LegacyFilterDict = Dict[str, List[LegacyFilterCondition]]

Filters = Union[Tuple[str, List["Filters"]], FilterCondition, LegacyFilterDict]

TextQueryOp = Literal["BM25"]
TextAggQueryOp = Literal["Sum"]

RankInputTextQuery = Union[
    Tuple[str, TextQueryOp, str],
    Tuple[TextAggQueryOp, Iterable["RankInputTextQuery"]],
]

AttributeOrdering = Union[Literal["asc"], Literal["desc"]]

RankInputOrderByAttribute = Tuple[str, AttributeOrdering]

# Uses tuples for nice typing, but also allows arrays for backwards compatibility
RankInput = Union[
    RankInputTextQuery,
    RankInputOrderByAttribute,
    List[Union[str, List[str]]],
]

ConsistencyDict = Dict[Literal["level"], Literal["strong", "eventual"]]
ConsistencyLevel = Literal["strong", "eventual"]

# Filter class alias for backwards compatibility
Filter = Dict[str, List[Tuple[str, Union[str, int, List[str], List[int]]]]]
# Op alias for backwards compatibility
Op = str


@dataclass
class VectorQuery:
    vector: Optional[List[float]] = None
    distance_metric: Optional[str] = None
    top_k: int = 10
    include_vectors: bool = False
    include_attributes: Optional[Union[List[str], bool]] = None
    filters: Optional[Filters] = None
    rank_by: Optional[RankInput] = None
    consistency: Optional[ConsistencyDict] = None

    @staticmethod
    def from_dict(source: dict) -> "VectorQuery":
        return VectorQuery(
            vector=source.get("vector"),
            distance_metric=source.get("distance_metric"),
            top_k=source.get("top_k", 10),  # Default to 10 if not provided
            include_vectors=source.get(
                "include_vectors", False
            ),  # Default to False if not provided
            include_attributes=source.get("include_attributes"),
            filters=source.get("filters"),
            rank_by=source.get("rank_by"),
            consistency=source.get("consistency"),
        )

    def __post_init__(self):
        if self.vector is not None:
            if "numpy" in sys.modules:
                # Check if it's a numpy array in a type-safe way
                numpy_module = sys.modules["numpy"]
                if hasattr(numpy_module, "ndarray"):
                    ndarray_type = getattr(numpy_module, "ndarray")
                    if isinstance(self.vector, ndarray_type):
                        # Now we can safely access ndim
                        if getattr(self.vector, "ndim", 0) != 1:
                            ndim = getattr(self.vector, "ndim", 0)
                            raise ValueError(
                                f"VectorQuery.vector must be a 1d array, got {ndim} dimensions"
                            )
            elif not isinstance(self.vector, list):
                raise ValueError(
                    "VectorQuery.vector must be a list, got:", type(self.vector)
                )
        if self.include_attributes is not None:
            if not isinstance(self.include_attributes, list) and not isinstance(
                self.include_attributes, bool
            ):
                raise ValueError(
                    "VectorQuery.include_attributes must be a list or bool, got:",
                    type(self.include_attributes),
                )
        if self.filters is not None:
            if (
                not isinstance(self.filters, dict)
                and not isinstance(self.filters, list)
                and not isinstance(self.filters, tuple)
            ):
                raise ValueError(
                    "VectorQuery.filters must be a dict, tuple, or list, got:",
                    type(self.filters),
                )
        if self.rank_by is not None:
            if not isinstance(self.rank_by, list) and not isinstance(
                self.rank_by, tuple
            ):
                raise ValueError(
                    "VectorQuery.rank_by must be a list or tuple, got:",
                    type(self.rank_by),
                )
            for item in self.rank_by:
                if (
                    not isinstance(item, str)
                    and not isinstance(item, list)
                    and not isinstance(item, tuple)
                ):
                    raise ValueError(
                        "VectorQuery.rank_by elements must be strings, tuples or lists, got:",
                        type(item),
                    )
        if self.consistency is not None:
            if (
                not isinstance(self.consistency, dict)
                or "level" not in self.consistency
            ):
                raise ValueError(
                    "VectorQuery.consistency must be a dict with a 'level' key"
                )
            if self.consistency["level"] not in ("strong", "eventual"):
                raise ValueError(
                    "VectorQuery.consistency level must be 'strong' or 'eventual', got:",
                    self.consistency["level"],
                )
