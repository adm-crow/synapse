import click

from . import __version__
from .pipeline import ingest, purge, query, reset, sources
from .sqlite_ingester import ingest_sqlite


@click.group()
@click.version_option(__version__, prog_name="synapse")
def cli() -> None:
    """synapse — local RAG: ingest files, query semantically."""


@cli.command(name="ingest")
@click.argument("source_dir", default="./docs")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
@click.option("--chunk-size", default=1000, show_default=True,
              help="Target characters per chunk.")
@click.option("--overlap", default=200, show_default=True,
              help="Character overlap between chunks.")
@click.option("--incremental", is_flag=True,
              help="Skip files whose content hasn't changed (SHA-256 check).")
@click.option("--chunking", default="word", show_default=True,
              type=click.Choice(["word", "sentence"]),
              help="Chunking strategy.")
def ingest_cmd(
    source_dir: str,
    db_path: str,
    collection: str,
    chunk_size: int,
    overlap: int,
    incremental: bool,
    chunking: str,
) -> None:
    """Ingest files from SOURCE_DIR into ChromaDB."""
    ingest(
        source_dir=source_dir,
        db_path=db_path,
        collection_name=collection,
        chunk_size=chunk_size,
        overlap=overlap,
        incremental=incremental,
        chunking=chunking,
    )


@cli.command(name="query")
@click.argument("text")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
@click.option("-n", "--n-results", default=5, show_default=True,
              help="Number of results to return.")
def query_cmd(text: str, db_path: str, collection: str, n_results: int) -> None:
    """Semantic search over the ChromaDB collection."""
    results = query(
        text=text,
        db_path=db_path,
        collection_name=collection,
        n_results=n_results,
    )
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        title = f" [{r['doc_title']}]" if r.get("doc_title") else ""
        click.echo(
            f"[{i}] score={r['score']:.3f}  {r['source']}{title}  chunk={r['chunk']}"
        )
        click.echo(f"    {r['text'][:200]}")
        click.echo()


@cli.command(name="sources")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
def sources_cmd(db_path: str, collection: str) -> None:
    """List all ingested source paths."""
    paths = sources(db_path=db_path, collection_name=collection)
    if not paths:
        click.echo("Collection is empty.")
        return
    for path in paths:
        click.echo(path)


@cli.command(name="purge")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
def purge_cmd(db_path: str, collection: str) -> None:
    """Remove chunks whose source file no longer exists on disk."""
    purge(db_path=db_path, collection_name=collection)


@cli.command(name="reset")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def reset_cmd(db_path: str, collection: str, yes: bool) -> None:
    """Wipe the entire ChromaDB collection (destructive)."""
    if not yes:
        click.confirm(
            f"Delete collection '{collection}' in '{db_path}'? This cannot be undone.",
            abort=True,
        )
    reset(db_path=db_path, collection_name=collection)
