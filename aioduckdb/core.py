# Copyright 2022 Amethyst Reese
# Copyright 2022 Salvador Pardiñas (port to DuckDB)
# Licensed under the MIT license

"""
Core implementation of aioduckdb proxies
"""

import asyncio
import logging
import duckdb
import sys
import warnings
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generator,
    Iterable,
    Optional,
    Type,
    Union,
    Tuple,
    TYPE_CHECKING
)
from warnings import warn

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from .context import contextmanager
from .cursor import Cursor
from .relation import Relation

if TYPE_CHECKING:
    import pandas
    import pyarrow

__all__ = ["connect", "Connection", "Cursor"]

LOG = logging.getLogger("aioduckdb")


# IsolationLevel = Optional[Literal["DEFERRED", "IMMEDIATE", "EXCLUSIVE"]]


def get_loop(future: asyncio.Future) -> asyncio.AbstractEventLoop:
    if sys.version_info >= (3, 7):
        return future.get_loop()
    else:
        return future._loop


class Connection(Thread):
    def __init__(
        self,
        connector: Callable[[], duckdb.DuckDBPyConnection],
        iter_chunk_size: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__()
        self._running = True
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connector = connector
        self._tx: Queue = Queue()
        self._iter_chunk_size = iter_chunk_size

        if loop is not None:
            warn(
                "aiosqlite.Connection no longer uses the `loop` parameter",
                DeprecationWarning,
            )

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise ValueError("no active connection")

        return self._connection

    # No automatic last_rowid, use an insert with `returning (rowid)`
    # additionally, rowid is currently broken: https://github.com/duckdb/duckdb/issues/4805 and sequences are recommended as an alternative
    def _execute_insert(
        self, sql: str, parameters: Iterable[Any]
    ) -> Union[Optional[Tuple[int]], object]:
        cursor = self._conn.execute(sql, parameters)
        return cursor.fetchone()

    def _execute_fetchall(
        self, sql: str, parameters: Iterable[Any]
    ) -> Iterable[Tuple]:
        cursor = self._conn.execute(sql, parameters)
        return cursor.fetchall()

    def run(self) -> None:
        """
        Execute function calls on a separate thread.

        :meta private:
        """
        while True:
            # Continues running until all queue items are processed,
            # even after connection is closed (so we can finalize all
            # futures)
            try:
                future, function = self._tx.get(timeout=0.1)
            except Empty:
                if self._running:
                    continue
                break
            try:
                LOG.debug("executing %s", function)
                result = function()
                LOG.debug("operation %s completed", function)

                def set_result(fut, result):
                    if not fut.done():
                        fut.set_result(result)

                get_loop(future).call_soon_threadsafe(set_result, future, result)
            except BaseException as e:
                LOG.debug("returning exception %s", e)

                def set_exception(fut, e):
                    if not fut.done():
                        fut.set_exception(e)

                get_loop(future).call_soon_threadsafe(set_exception, future, e)

    async def _execute(self, fn, *args, **kwargs):
        """Queue a function with the given arguments for execution."""
        if not self._running or not self._connection:
            raise ValueError("Connection closed")

        function = partial(fn, *args, **kwargs)
        future = asyncio.get_event_loop().create_future()

        self._tx.put_nowait((future, function))

        return await future

    async def _connect(self) -> "Connection":
        """Connect to the actual sqlite database."""
        if self._connection is None:
            try:
                future = asyncio.get_event_loop().create_future()
                self._tx.put_nowait((future, self._connector))
                self._connection = await future
            except Exception:
                self._running = False
                self._connection = None
                raise

        return self

    def __await__(self) -> Generator[Any, None, "Connection"]:
        self.start()
        return self._connect().__await__()

    async def __aenter__(self) -> "Connection":
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @contextmanager
    async def cursor(self) -> Cursor:
        """Create an aiosqlite cursor wrapping a sqlite3 cursor object (actually a duplicate of the connection, so make sure you're not trying to use registered tables between different connections)."""
        return Cursor(self, await self._execute(self._conn.cursor))

    @contextmanager
    async def query(self, query: str, alias: str = 'query_relation', parameters: Optional[Iterable[Any]] = None) -> Relation:
        """Create an aioduckdb relation wrapping a duckdb PyRelation object"""
        if parameters is None:
            parameters = []
        return Relation(self, await self._execute(self._conn.query, query, alias=alias, params=parameters))

    @contextmanager
    async def table(self, table_name: str) -> Relation:
        """Create a relation object for the name’d table"""
        return Relation(self, await self._execute(self._conn.table, table_name))

    @contextmanager
    async def values(self, values: Iterable) -> Relation:
        return Relation(self, await self._execute(self._conn.values, values))

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._execute(self._conn.commit)

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        await self._execute(self._conn.rollback)

    async def close(self) -> None:
        """Complete queued queries/cursors and close the connection."""
        try:
            await self._execute(self._conn.close)
        except Exception:
            LOG.info("exception occurred while closing connection")
            raise
        finally:
            self._running = False
            self._connection = None

    @contextmanager
    async def execute(self, sql: str, parameters: Optional[Iterable[Any]] = None) -> Cursor:
        """Helper to create a cursor and execute the given query.
        Huge warning: this function does not copy over registered objects and such. If you need those,
        use execute_on_self or as_cursor
        """
        if parameters is None:
            parameters = []
        cursor = await self.cursor()
        await cursor.execute(sql, parameters)
        return cursor

    @contextmanager
    async def execute_on_self(self, sql: str, parameters: Optional[Iterable[Any]] = None) -> Cursor:
        """Function to execute the given query on this connection and cast it to a cursor."""
        if parameters is None:
            parameters = []
        cursor = Cursor(self, self._conn)
        await cursor.execute(sql, parameters)
        return cursor

    @contextmanager
    async def as_cursor(self) -> Cursor:
        """Casts this connection to a cursor"""
        return Cursor(self, self._conn)

    @contextmanager
    async def execute_insert(
        self, sql: str, parameters: Iterable[Any] = None
    ) -> Optional[Tuple]:
        """Helper to insert and get the last_insert_rowid."""
        if parameters is None:
            parameters = []
        return await self._execute(self._execute_insert, sql, parameters)

    @contextmanager
    async def execute_fetchall(
        self, sql: str, parameters: Iterable[Any] = None
    ) -> Iterable[Tuple]:
        """Helper to execute a query and return all the data."""
        if parameters is None:
            parameters = []
        return await self._execute(self._execute_fetchall, sql, parameters)

    @contextmanager
    async def executemany(
        self, sql: str, parameters: Iterable[Iterable[Any]]
    ) -> Cursor:
        """Helper to create a cursor and execute the given multiquery."""
        cursor = await self._execute(self._conn.executemany, sql, parameters)
        return Cursor(self, cursor)

    @contextmanager
    async def from_csv_auto(
        self, file_name: str
    ) -> Relation:
        relation = await self._execute(self._conn.from_csv_auto, file_name)
        return Relation(self, relation)

    @contextmanager
    async def from_df(
        self, df: "pandas.DataFrame"
    ) -> Relation:
        relation = await self._execute(self._conn.from_df, df)
        return Relation(self, relation)

    @contextmanager
    async def from_arrow(
        self, arrow_object: object
    ) -> Relation:
        relation = await self._execute(self._conn.from_arrow, arrow_object)
        return Relation(self, relation)

    @contextmanager
    async def from_parquet(
        self, file_name: str, binary_as_string: bool = False
    ) -> Relation:
        relation = await self._execute(self._conn.from_parquet, file_name, binary_as_string=binary_as_string)
        return Relation(self, relation)

    async def register(
        self, view_name: str, python_object: object
    ) -> "Connection":
        await self._execute(self._conn.register, view_name, python_object)
        return self

    async def unregister(
        self, view_name: str
    ) -> "Connection":
        await self._execute(self._conn.unregister, view_name)
        return self

    # Apparently no equivalent to executescript? Verify API and compare to sqlite3's internals
    # @contextmanager
    # async def executescript(self, sql_script: str) -> Cursor:
    #     """Helper to create a cursor and execute a user script."""
    #     cursor = await self._execute(self._conn.executescript, sql_script)
    #     return Cursor(self, cursor)

    # DuckDB has very limited support for UDF and is not appliable yet
    # async def create_function(
    #     self, name: str, num_params: int, func: Callable, deterministic: bool = False
    # ) -> None:
    #     """
    #     Create user-defined function that can be later used
    #     within SQL statements. Must be run within the same thread
    #     that query executions take place so instead of executing directly
    #     against the connection, we defer this to `run` function.

    #     In Python 3.8 and above, if *deterministic* is true, the created
    #     function is marked as deterministic, which allows SQLite to perform
    #     additional optimizations. This flag is supported by SQLite 3.8.3 or
    #     higher, ``NotSupportedError`` will be raised if used with older
    #     versions.
    #     """
    #     if sys.version_info >= (3, 8):
    #         await self._execute(
    #             self._conn.create_function,
    #             name,
    #             num_params,
    #             func,
    #             deterministic=deterministic,
    #         )
    #     else:
    #         if deterministic:
    #             warnings.warn(
    #                 "Deterministic function support is only available on "
    #                 'Python 3.8+. Function "{}" will be registered as '
    #                 "non-deterministic as per SQLite defaults.".format(name)
    #             )

    #         await self._execute(self._conn.create_function, name, num_params, func)

    # No equivalent to in_process in DuckDB: autocommit mode executes inserts or updates instantly.
    # The only isolation level in DuckDB is Snapshot and there's no SQLite equivalent

    # No factories in DuckDB: proof of concept was developed and performance was worse
    # than python postprocessing
    
    async def load_extension(self, path: str):
        await self._execute(self._conn.load_extension, path)  # type: ignore

    # No progress handlers or trace_callback in DuckDB

    # Not supported in DuckDB's API but is available in the CLI, todo investigate
    # if it's easy to expose through the APIs

    # async def iterdump(self) -> AsyncIterator[str]:
    #     """
    #     Return an async iterator to dump the database in SQL text format.

    #     Example::

    #         async for line in db.iterdump():
    #             ...

    #     """
    #     dump_queue: Queue = Queue()

    #     def dumper():
    #         try:
    #             for line in self._conn.iterdump():
    #                 dump_queue.put_nowait(line)
    #             dump_queue.put_nowait(None)

    #         except Exception:
    #             LOG.exception("exception while dumping db")
    #             dump_queue.put_nowait(None)
    #             raise

    #     fut = self._execute(dumper)
    #     task = asyncio.ensure_future(fut)

    #     while True:
    #         try:
    #             line: Optional[str] = dump_queue.get_nowait()
    #             if line is None:
    #                 break
    #             yield line

    #         except Empty:
    #             if task.done():
    #                 LOG.warning("iterdump completed unexpectedly")
    #                 break

    #             await asyncio.sleep(0.01)

    #     await task

    # Backup is not supported by DuckDB, mostly due to being an OLAP oriented database
    # async def backup(
    #     self,
    #     target: Union["Connection", sqlite3.Connection],
    #     *,
    #     pages: int = 0,
    #     progress: Optional[Callable[[int, int, int], None]] = None,
    #     name: str = "main",
    #     sleep: float = 0.250
    # ) -> None:
    #     """
    #     Make a backup of the current database to the target database.

    #     Takes either a standard sqlite3 or aiosqlite Connection object as the target.
    #     """
    #     if sys.version_info < (3, 7):
    #         raise RuntimeError("backup() method is only available on Python 3.7+")

    #     if isinstance(target, Connection):
    #         target = target._conn

    #     await self._execute(
    #         self._conn.backup,
    #         target,
    #         pages=pages,
    #         progress=progress,
    #         name=name,
    #         sleep=sleep,
    #     )


def connect(
    database: Union[str, Path],
    *,
    iter_chunk_size=64,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    **kwargs: Any
) -> Connection:
    """Create and return a connection proxy to the sqlite database."""

    if loop is not None:
        warn(
            "aiosqlite.connect() no longer uses the `loop` parameter",
            DeprecationWarning,
        )

    def connector() -> duckdb.DuckDBPyConnection:
        if isinstance(database, str):
            loc = database
        elif isinstance(database, bytes):
            loc = database.decode("utf-8")
        else:
            loc = str(database)

        return duckdb.connect(loc, **kwargs)

    return Connection(connector, iter_chunk_size)
