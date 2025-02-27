import asyncio
from dataclasses import dataclass, field
import json
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, overload

import iso8601

import turbopuffer as tpuf
from turbopuffer.backend import Backend
from turbopuffer.error import APIError
from turbopuffer.query import ConsistencyDict, Filters, RankInput, VectorQuery
from turbopuffer.vectors import (
    Cursor,
    VectorColumns,
    VectorResult,
    VectorRow,
    batch_iter,
)

CmekDict = Dict[Literal["key_name"], str]
EncryptionDict = Dict[Literal["cmek"], CmekDict]


@dataclass(frozen=True)
class FullTextSearchParams:
    """
    Used for configuring BM25 full-text indexing for a given attribute.
    """

    language: str
    stemming: bool
    remove_stopwords: bool
    case_sensitive: bool

    def __init__(
        self,
        language: str,
        stemming: bool,
        remove_stopwords: bool,
        case_sensitive: bool,
    ):
        self.language = language
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self.case_sensitive = case_sensitive

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "stemming": self.stemming,
            "remove_stopwords": self.remove_stopwords,
            "case_sensitive": self.case_sensitive,
        }


@dataclass(frozen=True)
class AttributeSchema:
    """
    The schema for a particular attribute within a namespace.
    """

    type: str  # one of: 'string', 'uint', '[]string', '[]uint'
    filterable: bool
    full_text_search: Optional[FullTextSearchParams] = None

    def __init__(
        self,
        type: str,
        filterable: bool,
        full_text_search: Optional[FullTextSearchParams] = None,
    ):
        self.type = type
        self.filterable = filterable
        self.full_text_search = full_text_search

    def as_dict(self) -> dict:
        result = {
            "type": self.type,
            "filterable": self.filterable,
        }
        if self.full_text_search:
            result["full_text_search"] = self.full_text_search.as_dict()
        return result


# Type alias for a namespace schema
NamespaceSchema = Dict[str, AttributeSchema]


def parse_namespace_schema(data: dict) -> NamespaceSchema:
    namespace_schema = {}
    for key, value in data.items():
        fts_params = value.get("full_text_search")
        fts_instance = None
        if fts_params:
            fts_instance = FullTextSearchParams(
                language=fts_params["language"],
                stemming=fts_params["stemming"],
                remove_stopwords=fts_params["remove_stopwords"],
                case_sensitive=fts_params["case_sensitive"],
            )
        attribute_schema = AttributeSchema(
            type=value["type"],
            filterable=value["filterable"],
            full_text_search=fts_instance,
        )
        namespace_schema[key] = attribute_schema
    return namespace_schema


@dataclass
class Namespace:
    """
    The Namespace type represents a set of vectors stored in turbopuffer.
    Within a namespace, vectors are uniquely referred to by their ID.
    All vectors in a namespace must have the same dimensions.
    """

    name: str
    backend: Backend

    metadata: Optional[dict] = None

    def __init__(
        self, name: str, api_key: Optional[str] = None, headers: Optional[dict] = None
    ):
        """
        Creates a new turbopuffer.Namespace object for querying the turbopuffer API.

        This function does not make any API calls on its own.

        Specifying an api_key here will override the global configuration for API calls to this namespace.
        """
        self.name = name
        self.backend = Backend(api_key, headers)

    def __str__(self) -> str:
        return f"tpuf-namespace:{self.name}"

    def __eq__(self, other):
        if isinstance(other, Namespace):
            return self.name == other.name and self.backend == other.backend
        else:
            return False

    async def arefresh_metadata(self):
        """
        Asynchronous version of refresh_metadata
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, method="HEAD"
        )
        status_code = response.get("status_code")
        if status_code == 200:
            headers = response.get("headers", dict())

            # Record any metadata we got in the HEAD request.
            self.metadata = {
                # This must be explicit since the API might return a 200 OK, but the embedding doesn't exist.
                "exists": headers.get("X-Exists", "false").lower() == "true",
            }

            # Parse created_at if available
            created_at_str = headers.get("X-Created-At")
            if created_at_str:
                try:
                    self.metadata["created_at"] = iso8601.parse_date(created_at_str)
                except (ValueError, TypeError):
                    pass

            # Parse dimensions and approx_count if available
            dimensions_str = headers.get("X-Dimensions")
            if dimensions_str:
                try:
                    self.metadata["dimensions"] = int(dimensions_str)
                except (ValueError, TypeError):
                    pass
            approx_count_str = headers.get("X-Approx-Count")
            if approx_count_str:
                try:
                    self.metadata["approx_count"] = int(approx_count_str)
                except (ValueError, TypeError):
                    pass
        else:
            raise APIError(
                response.get("status_code", 500),
                "Unexpected status code",
                str(response.get("content") or ""),
            )

    def refresh_metadata(self):
        """
        Refresh metadata from the API
        """
        return asyncio.run(self.arefresh_metadata())

    async def aexists(self) -> bool:
        """
        Asynchronous version to check if the namespace exists
        """
        # Always refresh the exists check since metadata from namespaces() might be delayed.
        await self.arefresh_metadata()
        if self.metadata is None:
            return False
        return self.metadata["exists"]

    def exists(self) -> bool:
        """
        Check if the namespace exists
        """
        return asyncio.run(self.aexists())

    async def adimensions(self) -> int:
        """
        Asynchronous version to get the dimensions of vectors in the namespace
        """
        if self.metadata is None or "dimensions" not in self.metadata:
            await self.arefresh_metadata()
        if self.metadata is None:
            return 0
        return self.metadata.get("dimensions", 0)

    def dimensions(self) -> int:
        """
        Get the dimensions of vectors in the namespace
        """
        return asyncio.run(self.adimensions())

    async def aapprox_count(self) -> int:
        """
        Asynchronous version to get the approximate count of vectors in the namespace
        """
        if self.metadata is None or "approx_count" not in self.metadata:
            await self.arefresh_metadata()
        if self.metadata is None:
            return 0
        return self.metadata.get("approx_count", 0)

    def approx_count(self) -> int:
        """
        Get the approximate count of vectors in the namespace
        """
        return asyncio.run(self.aapprox_count())

    async def acreated_at(self) -> Optional[datetime]:
        """
        Asynchronous version to get when the namespace was created
        """
        if self.metadata is None or "created_at" not in self.metadata:
            await self.arefresh_metadata()
        if self.metadata is None:
            return None
        return self.metadata.get("created_at", None)

    def created_at(self) -> Optional[datetime]:
        """
        Get when the namespace was created
        """
        return asyncio.run(self.acreated_at())

    async def aschema(self) -> NamespaceSchema:
        """
        Asynchronous version to get the current schema for the namespace
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, "schema", method="GET"
        )
        return parse_namespace_schema(response["content"])

    def schema(self) -> NamespaceSchema:
        """
        Returns the current schema for the namespace.
        """
        return asyncio.run(self.aschema())

    async def aupdate_schema(self, schema_updates: NamespaceSchema):
        """
        Asynchronous version to update the schema for the namespace
        """
        # Create a dictionary payload instead of encoding it as JSON bytes
        schema_dict = {key: value.as_dict() for key, value in schema_updates.items()}
        response = await self.backend.amake_api_request(
            "namespaces", self.name, "schema", method="POST", payload=schema_dict
        )
        return parse_namespace_schema(response["content"])

    def update_schema(self, schema_updates: NamespaceSchema):
        """
        Update the schema for the namespace.
        This will add or modify attributes, but not remove them.
        """
        return asyncio.run(self.aupdate_schema(schema_updates))

    async def acopy_from_namespace(self, source_namespace: str):
        """
        Asynchronous version to create a copy of another namespace
        """
        payload = {"copy_from_namespace": source_namespace}
        response = await self.backend.amake_api_request(
            "namespaces", self.name, payload=payload
        )
        assert response.get("content", dict()).get("status", "") == "OK", (
            f"Invalid copy_from_namespace() response: {response}"
        )

    def copy_from_namespace(self, source_namespace: str):
        """
        Create a copy of another namespace. This will copy all vectors and the schema.
        """
        return asyncio.run(self.acopy_from_namespace(source_namespace))

    @overload
    async def aupsert(
        self,
        ids: Union[List[int], List[str]],
        vectors: List[List[float]],
        attributes: Optional[Dict[str, List[Optional[Union[str, int]]]]] = None,
        schema: Optional[Dict] = None,
        distance_metric: Optional[str] = None,
        encryption: Optional[EncryptionDict] = None,
    ) -> None: ...

    @overload
    async def aupsert(
        self,
        data: Union[dict, VectorColumns],
        distance_metric: Optional[str] = None,
        schema: Optional[Dict] = None,
        encryption: Optional[EncryptionDict] = None,
    ) -> None: ...

    @overload
    async def aupsert(
        self,
        data: Union[Iterable[dict], Iterable[VectorRow]],
        distance_metric: Optional[str] = None,
        schema: Optional[Dict] = None,
        encryption: Optional[EncryptionDict] = None,
    ) -> None: ...

    @overload
    async def aupsert(
        self,
        data: VectorResult,
        distance_metric: Optional[str] = None,
        schema: Optional[Dict] = None,
        encryption: Optional[EncryptionDict] = None,
    ) -> None: ...

    async def aupsert(
        self,
        data=None,
        ids=None,
        vectors=None,
        attributes=None,
        schema=None,
        distance_metric=None,
        encryption=None,
    ) -> None:
        """
        Asynchronous version to insert or update vectors in the namespace
        """
        if data is None and ids is None:
            raise ValueError(
                "upsert() must take either data= or (ids=, vectors=) arguments"
            )
        elif (ids is None and vectors is not None) or (
            vectors is None and ids is not None
        ):
            raise ValueError("upsert() with ids= must also have vectors=")
        elif (ids is not None and attributes is None) or (
            attributes is not None and schema is None
        ):
            # Offset arguments to handle positional arguments case with no data field.
            return await self.aupsert(
                VectorColumns(ids=data, vectors=ids, attributes=vectors),
                schema=attributes,
                distance_metric=distance_metric,
                encryption=encryption,
            )
        elif isinstance(data, VectorColumns):
            # "if None in data.vectors:" is not supported because data.vectors might be a list of np.ndarray
            # None == pd.ndarray is an ambiguous comparison in this case.

            if len(data.vectors) > 0 and None in data.vectors:
                raise ValueError("VectorColumns.vectors must not contain None values")

            payload = {
                "ids": data.ids,
                "vectors": data.vectors,
            }
            if data.attributes:
                payload["attributes"] = data.attributes

            if schema:
                payload["schema"] = schema

            if distance_metric:
                payload["distance_metric"] = distance_metric

            if encryption:
                payload["encryption"] = encryption

            response = await self.backend.amake_api_request(
                "namespaces", self.name, payload=payload
            )

            assert response.get("content", dict()).get("status", "") == "OK", (
                f"Invalid upsert() response: {response}"
            )
            self.metadata = None  # Invalidate cached metadata

        elif isinstance(data, list):
            if isinstance(data[0], dict):
                return await self.aupsert(
                    VectorColumns.from_rows(data),
                    schema=schema,
                    distance_metric=distance_metric,
                    encryption=encryption,
                )
            elif isinstance(data[0], VectorRow):
                return await self.aupsert(
                    VectorColumns.from_rows(data),
                    schema=schema,
                    distance_metric=distance_metric,
                    encryption=encryption,
                )
            elif isinstance(data[0], VectorColumns):
                for columns in data:
                    await self.aupsert(
                        columns,
                        schema=schema,
                        distance_metric=distance_metric,
                        encryption=encryption,
                    )
                return
            else:
                raise ValueError(
                    f"upsert() list type must be of dict, VectorRow, or VectorColumns: {type(data[0])}"
                )
        elif isinstance(data, dict):
            if "id" in data:
                # Convert a dict to a VectorRow
                if not "vector" in data:
                    raise ValueError(
                        "upsert() should be called on a list of vectors, got single vector."
                    )
            elif "ids" in data:
                return await self.aupsert(
                    VectorColumns.from_dict(data),
                    schema=data.get("schema", None),
                    distance_metric=distance_metric,
                    encryption=encryption,
                )
            else:
                raise ValueError("Provided dict is missing ids.")
        elif "pandas" in sys.modules and isinstance(
            data, sys.modules["pandas"].DataFrame
        ):
            cols = list(data.columns)
            if "id" not in cols or "vector" not in cols:
                missing = []
                if "id" not in cols:
                    missing.append("id")
                if "vector" not in cols:
                    missing.append("vector")
                raise ValueError(
                    f'DataFrame must have columns "id" and "vector", but it only contains: {cols}. Missing: {missing}'
                )
            # todo: optimize this to avoid copying the dataframe
            return await self.aupsert(
                VectorColumns.from_rows(
                    [
                        VectorRow(data.iloc[i]["id"], data.iloc[i]["vector"])
                        for i in range(len(data.index))
                    ]
                ),
                schema=schema,
                distance_metric=distance_metric,
                encryption=encryption,
            )
        else:
            raise ValueError(f"upsert() input type is not supported: {type(data)}")

    def upsert(
        self,
        data=None,
        ids=None,
        vectors=None,
        attributes=None,
        schema=None,
        distance_metric=None,
        encryption=None,
    ) -> None:
        """
        Insert or update vectors in the namespace.
        """
        return asyncio.run(
            self.aupsert(
                data=data,
                ids=ids,
                vectors=vectors,
                attributes=attributes,
                schema=schema,
                distance_metric=distance_metric,
                encryption=encryption,
            )
        )

    async def adelete(self, ids: Union[int, str, List[int], List[str]]) -> None:
        """
        Asynchronous version to delete vectors from the namespace
        """
        if isinstance(ids, int) or isinstance(ids, str):
            response = await self.backend.amake_api_request(
                "namespaces",
                self.name,
                payload={
                    "ids": [ids],
                    "vectors": [None],
                },
            )
        elif isinstance(ids, list):
            response = await self.backend.amake_api_request(
                "namespaces",
                self.name,
                payload={
                    "ids": ids,
                    "vectors": [None] * len(ids),
                },
            )
        else:
            raise ValueError(f"Invalid type for ids: {type(ids)}")

        assert response.get("content", dict()).get("status", "") == "OK", (
            f"Invalid delete() response: {response}"
        )
        self.metadata = None  # Invalidate cached metadata

    def delete(self, ids: Union[int, str, List[int], List[str]]) -> None:
        """
        Delete vectors from the namespace by their IDs.
        """
        return asyncio.run(self.adelete(ids))

    async def adelete_by_filter(self, filters: Filters) -> int:
        """
        Asynchronous version to delete vectors that match the filter
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, payload={"delete_by_filter": filters}
        )
        response_content = response.get("content", dict())
        assert response_content.get("status", "") == "ok", (
            f"Invalid delete_by_filter() response: {response}"
        )
        self.metadata = None  # Invalidate cached metadata
        return response_content.get("deleted_count", 0)

    def delete_by_filter(self, filters: Filters) -> int:
        """
        Delete vectors that match the filter.
        Returns the number of vectors deleted.
        """
        return asyncio.run(self.adelete_by_filter(filters))

    @overload
    async def aquery(
        self,
        vector: Optional[List[float]] = None,
        distance_metric: Optional[str] = None,
        top_k: int = 10,
        include_vectors: bool = False,
        include_attributes: Optional[Union[List[str], bool]] = None,
        filters: Optional[Filters] = None,
        rank_by: Optional[RankInput] = None,
        consistency: Optional[ConsistencyDict] = None,
    ) -> VectorResult: ...

    @overload
    async def aquery(self, query_data: VectorQuery) -> VectorResult: ...

    @overload
    async def aquery(self, query_data: dict) -> VectorResult: ...

    async def aquery(
        self,
        query_data=None,
        vector=None,
        distance_metric=None,
        top_k=None,
        include_vectors=None,
        include_attributes=None,
        filters=None,
        rank_by=None,
        consistency=None,
    ) -> VectorResult:
        """
        Asynchronous version to query vectors in the namespace
        """
        if query_data is None:
            return await self.aquery(
                VectorQuery(
                    vector=vector,
                    distance_metric=distance_metric,
                    top_k=top_k or 10,
                    include_vectors=include_vectors
                    if include_vectors is not None
                    else False,
                    include_attributes=include_attributes,
                    filters=filters,
                    rank_by=rank_by,
                    consistency=consistency,
                )
            )
        if not isinstance(query_data, VectorQuery):
            if isinstance(query_data, dict):
                query_data = VectorQuery.from_dict(query_data)
            else:
                raise ValueError(
                    f"query() input type must be compatible with turbopuffer.VectorQuery: {type(query_data)}"
                )

        response = await self.backend.amake_api_request(
            "namespaces", self.name, "query", payload=query_data.__dict__
        )
        result = VectorResult(response.get("content", dict()), namespace=self)
        result.performance = response.get("performance")
        return result

    def query(
        self,
        query_data=None,
        vector=None,
        distance_metric=None,
        top_k=None,
        include_vectors=None,
        include_attributes=None,
        filters=None,
        rank_by=None,
        consistency=None,
    ) -> VectorResult:
        """
        Query vectors in the namespace.
        """
        return asyncio.run(
            self.aquery(
                query_data=query_data,
                vector=vector,
                distance_metric=distance_metric,
                top_k=top_k,
                include_vectors=include_vectors,
                include_attributes=include_attributes,
                filters=filters,
                rank_by=rank_by,
                consistency=consistency,
            )
        )

    async def avectors(self, cursor: Optional[Cursor] = None) -> VectorResult:
        """
        Asynchronous version to get all vectors in the namespace
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, query={"cursor": cursor}
        )
        content = response.get("content", dict())
        next_cursor = content.pop("next_cursor", None)
        result = VectorResult(content, namespace=self, next_cursor=next_cursor)
        result.performance = response.get("performance")
        return result

    def vectors(self, cursor: Optional[Cursor] = None) -> VectorResult:
        """
        Get all the vectors in the namespace.
        """
        return asyncio.run(self.avectors(cursor))

    async def adelete_all_indexes(self) -> None:
        """
        Asynchronous version to delete all indexes in the namespace
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, "index", method="DELETE"
        )
        assert response.get("content", dict()).get("status", "") == "ok", (
            f"Invalid delete_all_indexes() response: {response}"
        )

    def delete_all_indexes(self) -> None:
        """
        Delete all indexes in the namespace.
        """
        return asyncio.run(self.adelete_all_indexes())

    async def adelete_all(self) -> None:
        """
        Asynchronous version to delete all vectors in the namespace
        """
        response = await self.backend.amake_api_request(
            "namespaces", self.name, method="DELETE"
        )
        assert response.get("content", dict()).get("status", "") == "ok", (
            f"Invalid delete_all() response: {response}"
        )
        self.metadata = None  # Invalidate cached metadata

    def delete_all(self) -> None:
        """
        Delete all vectors in the namespace.
        """
        return asyncio.run(self.adelete_all())

    async def arecall(self, num=20, top_k=10) -> float:
        """
        Asynchronous version to test the recall of the namespace
        """
        response = await self.backend.amake_api_request(
            "namespaces",
            self.name,
            "_debug",
            "recall",
            query={"num": num, "top_k": top_k},
        )
        content = response.get("content", dict())
        assert "avg_recall" in content, f"Invalid recall() response: {response}"
        return float(content.get("avg_recall"))

    def recall(self, num=20, top_k=10) -> float:
        """
        Test the recall of the namespace using brute force search as ground truth.
        """
        return asyncio.run(self.arecall(num, top_k))


@dataclass
class NamespaceIterator:
    """
    The VectorResult type represents a set of vectors that are the result of a query.

    A VectorResult can be treated as either a lazy iterator or a list by the user.
    Reading the length of the result will internally buffer the full result.
    """

    backend: Backend
    namespaces: List[Namespace] = field(default_factory=list)
    index: int = -1
    offset: int = 0
    next_cursor: Optional[Cursor] = None

    def __init__(
        self,
        backend: Backend,
        initial_set: Union[List[Namespace], List[dict]] = [],
        next_cursor: Optional[Cursor] = None,
    ):
        self.backend = backend
        self.index = -1
        self.offset = 0
        self.next_cursor = next_cursor

        if len(initial_set):
            if isinstance(initial_set[0], Namespace):
                self.namespaces = initial_set
            else:
                self.namespaces = NamespaceIterator.load_namespaces(
                    backend.api_key, initial_set
                )

    def load_namespaces(
        api_key: Optional[str], initial_set: List[dict]
    ) -> List[Namespace]:
        output = []
        for input in initial_set:
            ns = tpuf.Namespace(input["id"], api_key=api_key)
            ns.metadata = {
                "exists": True,
            }
            output.append(ns)

        return output

    def __str__(self) -> str:
        str_list = [ns.name for ns in self.namespaces]
        if not self.next_cursor and self.offset == 0:
            return str(str_list)
        else:
            return (
                "NamespaceIterator("
                f"offset={self.offset}, "
                f"next_cursor='{self.next_cursor}', "
                f"namespaces={str_list})"
            )

    def __len__(self) -> int:
        assert self.offset == 0, "Can't call len(NamespaceIterator) after iterating"
        assert self.index == -1, "Can't call len(NamespaceIterator) after iterating"
        if not self.next_cursor:
            return len(self.namespaces)
        else:
            it = iter(self)
            self.namespaces = [next for next in it]
            self.offset = 0
            self.index = -1
            self.next_cursor = None
            return len(self.namespaces)

    def __getitem__(self, index) -> VectorRow:
        if index >= len(self.namespaces) and self.next_cursor:
            it = iter(self)
            self.namespaces = [next for next in it]
            self.offset = 0
            self.index = -1
            self.next_cursor = None
        return self.namespaces[index]

    def __iter__(self) -> "NamespaceIterator":
        assert self.offset == 0, "Can't iterate over NamespaceIterator multiple times"
        return NamespaceIterator(self.backend, self.namespaces, self.next_cursor)

    def __next__(self):
        if self.index + 1 < len(self.namespaces):
            self.index += 1
            return self.namespaces[self.index]
        elif self.next_cursor is None:
            raise StopIteration
        else:
            response = self.backend.make_api_request(
                "namespaces", query={"cursor": self.next_cursor}
            )
            content = response.get("content", dict())
            self.offset += len(self.namespaces)
            self.index = -1
            self.next_cursor = content.pop("next_cursor", None)
            self.namespaces = NamespaceIterator.load_namespaces(
                self.backend.api_key, content.pop("namespaces", list())
            )
            return self.__next__()


async def anamespaces(api_key: Optional[str] = None) -> Iterable[Namespace]:
    """
    Asynchronous version of namespaces() that returns an iterator of all namespaces.
    """
    backend = Backend(api_key)
    response = await backend.amake_api_request("namespaces")
    content = response.get("content", dict())
    next_cursor = content.pop("next_cursor", None)
    return NamespaceIterator(backend, content.pop("namespaces", list()), next_cursor)


def namespaces(api_key: Optional[str] = None) -> Iterable[Namespace]:
    """
    Returns an iterator of all namespaces.
    """
    return asyncio.run(anamespaces(api_key))
