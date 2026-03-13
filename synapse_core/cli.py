import click

from . import __version__
from .pipeline import ingest, purge, query, reset, sources
from .sqlite_ingester import ingest_sqlite
from .ai import PROVIDERS, DEFAULT_MODELS, detect_provider, generate_answer


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
    try:
        ingest(
            source_dir=source_dir,
            db_path=db_path,
            collection_name=collection,
            chunk_size=chunk_size,
            overlap=overlap,
            incremental=incremental,
            chunking=chunking,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command(name="ingest-sqlite")
@click.argument("db_path")
@click.option("--table", required=True, help="Table name to ingest.")
@click.option("--db", "chroma_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
@click.option("--chunk-size", default=1000, show_default=True,
              help="Target characters per chunk.")
@click.option("--overlap", default=200, show_default=True,
              help="Character overlap between chunks.")
@click.option("--chunking", default="word", show_default=True,
              type=click.Choice(["word", "sentence"]),
              help="Chunking strategy.")
def ingest_sqlite_cmd(
    db_path: str,
    table: str,
    chroma_path: str,
    collection: str,
    chunk_size: int,
    overlap: int,
    chunking: str,
) -> None:
    """Ingest records from a SQLite DB_PATH table into ChromaDB."""
    try:
        ingest_sqlite(
            db_path=db_path,
            table=table,
            chroma_path=chroma_path,
            collection_name=collection,
            chunk_size=chunk_size,
            overlap=overlap,
            chunking=chunking,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command(name="query")
@click.argument("text")
@click.option("--db", "db_path", default="./synapse_db", show_default=True,
              help="ChromaDB persistence path.")
@click.option("--collection", default="synapse", show_default=True,
              help="Collection name.")
@click.option("-n", "--n-results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("--ai", "use_ai", is_flag=True,
              help="Generate an AI answer from the retrieved chunks.")
@click.option("--provider", default=None,
              type=click.Choice(list(PROVIDERS), case_sensitive=False),
              help="LLM provider (anthropic, openai, ollama). Auto-detected if omitted.")
@click.option("--model", default=None,
              help="Model name override (e.g. gpt-4o, llama3).")
def query_cmd(
    text: str,
    db_path: str,
    collection: str,
    n_results: int,
    use_ai: bool,
    provider: str | None,
    model: str | None,
) -> None:
    """Semantic search over the ChromaDB collection.

    Add --ai to generate a synthesized answer via an LLM.
    Provider is auto-detected from ANTHROPIC_API_KEY / OPENAI_API_KEY / Ollama.
    """
    try:
        results = query(
            text=text,
            db_path=db_path,
            collection_name=collection,
            n_results=n_results,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if not results:
        click.echo("No results found.")
        return

    if use_ai:
        # Resolve provider
        resolved = provider or detect_provider()
        if resolved is None:
            click.echo(
                "Error: no AI provider detected.\n"
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or start 'ollama serve'.\n"
                "You can also pass --provider explicitly.",
                err=True,
            )
            raise SystemExit(1)

        resolved_model = model or DEFAULT_MODELS.get(resolved, "")
        context = "\n\n".join(r["text"] for r in results)

        try:
            answer = generate_answer(
                question=text,
                context=context,
                provider=resolved,
                model=resolved_model,
            )
        except (ImportError, RuntimeError, ValueError) as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

        click.echo(f"\nAnswer  [{resolved} / {resolved_model}]")
        click.echo("─" * 60)
        click.echo(answer)
        click.echo("\nSources")
        click.echo("─" * 60)
        seen: set[str] = set()
        for r in results:
            src = r["source"]
            if src not in seen:
                title = f"  [{r['doc_title']}]" if r.get("doc_title") else ""
                click.echo(f"  [{r['score']:.2f}] {src}{title}")
                seen.add(src)
        click.echo()
        return

    # Default: raw results
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
