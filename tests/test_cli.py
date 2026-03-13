from unittest.mock import patch

from click.testing import CliRunner

from synapse_core.cli import cli


@patch("synapse_core.cli.ingest")
def test_cli_ingest_default_args(mock_ingest):
    result = CliRunner().invoke(cli, ["ingest", "./docs"])
    assert result.exit_code == 0
    mock_ingest.assert_called_once()
    call_kwargs = mock_ingest.call_args.kwargs
    assert call_kwargs["source_dir"] == "./docs"
    assert call_kwargs["incremental"] is False
    assert call_kwargs["chunking"] == "word"


@patch("synapse_core.cli.ingest")
def test_cli_ingest_incremental_flag(mock_ingest):
    result = CliRunner().invoke(cli, ["ingest", "./docs", "--incremental"])
    assert result.exit_code == 0
    assert mock_ingest.call_args.kwargs["incremental"] is True


@patch("synapse_core.cli.ingest")
def test_cli_ingest_sentence_chunking(mock_ingest):
    result = CliRunner().invoke(cli, ["ingest", "./docs", "--chunking", "sentence"])
    assert result.exit_code == 0
    assert mock_ingest.call_args.kwargs["chunking"] == "sentence"


@patch("synapse_core.cli.query")
def test_cli_query_returns_results(mock_query):
    mock_query.return_value = [
        {
            "text": "Refunds are accepted within 30 days.",
            "source": "/docs/policy.txt",
            "score": 0.91,
            "chunk": 0,
            "doc_title": "Policy",
            "doc_author": "",
            "doc_created": "",
        }
    ]
    result = CliRunner().invoke(cli, ["query", "refund policy"])
    assert result.exit_code == 0
    assert "0.910" in result.output
    assert "policy.txt" in result.output
    assert "Refunds are accepted" in result.output


@patch("synapse_core.cli.query")
def test_cli_query_empty_results(mock_query):
    mock_query.return_value = []
    result = CliRunner().invoke(cli, ["query", "nothing"])
    assert result.exit_code == 0
    assert "No results" in result.output


@patch("synapse_core.cli.sources")
def test_cli_sources_lists_paths(mock_sources):
    mock_sources.return_value = ["/docs/a.txt", "/docs/b.pdf"]
    result = CliRunner().invoke(cli, ["sources"])
    assert result.exit_code == 0
    assert "/docs/a.txt" in result.output
    assert "/docs/b.pdf" in result.output


@patch("synapse_core.cli.sources")
def test_cli_sources_empty(mock_sources):
    mock_sources.return_value = []
    result = CliRunner().invoke(cli, ["sources"])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()


@patch("synapse_core.cli.purge")
def test_cli_purge(mock_purge):
    result = CliRunner().invoke(cli, ["purge"])
    assert result.exit_code == 0
    mock_purge.assert_called_once()


@patch("synapse_core.cli.reset")
def test_cli_reset_with_yes_flag(mock_reset):
    result = CliRunner().invoke(cli, ["reset", "--yes"])
    assert result.exit_code == 0
    mock_reset.assert_called_once()


@patch("synapse_core.cli.reset")
def test_cli_reset_aborts_without_confirmation(mock_reset):
    """reset without --yes must prompt; answering 'n' must abort."""
    result = CliRunner().invoke(cli, ["reset"], input="n\n")
    assert result.exit_code != 0
    mock_reset.assert_not_called()


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.5.2" in result.output


# --- ingest-sqlite ---

@patch("synapse_core.cli.ingest_sqlite")
def test_cli_ingest_sqlite_basic(mock_ingest_sqlite):
    result = CliRunner().invoke(cli, ["ingest-sqlite", "./data.db", "--table", "articles"])
    assert result.exit_code == 0
    mock_ingest_sqlite.assert_called_once()
    kw = mock_ingest_sqlite.call_args.kwargs
    assert kw["db_path"] == "./data.db"
    assert kw["table"] == "articles"
    assert kw["chunking"] == "word"


@patch("synapse_core.cli.ingest_sqlite")
def test_cli_ingest_sqlite_missing_table_errors(mock_ingest_sqlite):
    """--table is required; omitting it must exit non-zero."""
    result = CliRunner().invoke(cli, ["ingest-sqlite", "./data.db"])
    assert result.exit_code != 0
    mock_ingest_sqlite.assert_not_called()


# --- --ai flag ---

_FAKE_RESULT = [
    {
        "text": "Refunds are accepted within 30 days.",
        "source": "/docs/policy.txt",
        "score": 0.91,
        "chunk": 0,
        "doc_title": "Policy",
        "doc_author": "",
        "doc_created": "",
    }
]


@patch("synapse_core.cli.generate_answer", return_value="You can return within 30 days.")
@patch("synapse_core.cli.detect_provider", return_value="anthropic")
@patch("synapse_core.cli.query")
def test_cli_query_ai_flag_shows_answer(mock_query, mock_detect, mock_generate):
    mock_query.return_value = _FAKE_RESULT
    result = CliRunner().invoke(cli, ["query", "refund policy", "--ai"])
    assert result.exit_code == 0
    assert "You can return within 30 days." in result.output
    assert "anthropic" in result.output
    assert "Sources" in result.output


@patch("synapse_core.cli.generate_answer", return_value="Answer here.")
@patch("synapse_core.cli.detect_provider", return_value="openai")
@patch("synapse_core.cli.query")
def test_cli_query_ai_explicit_provider(mock_query, mock_detect, mock_generate):
    mock_query.return_value = _FAKE_RESULT
    result = CliRunner().invoke(cli, ["query", "refund", "--ai", "--provider", "openai"])
    assert result.exit_code == 0
    mock_generate.assert_called_once()
    assert mock_generate.call_args.kwargs["provider"] == "openai"


@patch("synapse_core.cli.generate_answer", return_value="Answer here.")
@patch("synapse_core.cli.detect_provider", return_value="ollama")
@patch("synapse_core.cli.query")
def test_cli_query_ai_model_override(mock_query, mock_detect, mock_generate):
    mock_query.return_value = _FAKE_RESULT
    result = CliRunner().invoke(cli, ["query", "refund", "--ai", "--model", "mistral"])
    assert result.exit_code == 0
    assert mock_generate.call_args.kwargs["model"] == "mistral"


@patch("synapse_core.cli.detect_provider", return_value=None)
@patch("synapse_core.cli.query")
def test_cli_query_ai_no_provider_detected(mock_query, mock_detect):
    mock_query.return_value = _FAKE_RESULT
    result = CliRunner().invoke(cli, ["query", "refund", "--ai"])
    assert result.exit_code != 0
    assert "no AI provider" in result.output


@patch("synapse_core.cli.generate_answer", side_effect=ImportError("pip install anthropic"))
@patch("synapse_core.cli.detect_provider", return_value="anthropic")
@patch("synapse_core.cli.query")
def test_cli_query_ai_missing_sdk_shows_error(mock_query, mock_detect, mock_generate):
    mock_query.return_value = _FAKE_RESULT
    result = CliRunner().invoke(cli, ["query", "refund", "--ai"])
    assert result.exit_code != 0
    assert "Error:" in result.output


# --- error handling ---

@patch("synapse_core.cli.ingest")
def test_cli_ingest_file_not_found_shows_error(mock_ingest):
    mock_ingest.side_effect = FileNotFoundError("Source directory not found: ./missing")
    result = CliRunner().invoke(cli, ["ingest", "./missing"])
    assert result.exit_code != 0
    assert "Error:" in result.output


@patch("synapse_core.cli.query")
def test_cli_query_collection_not_found_shows_error(mock_query):
    mock_query.side_effect = ValueError("Collection 'synapse' not found — run ingest() first.")
    result = CliRunner().invoke(cli, ["query", "test"])
    assert result.exit_code != 0
    assert "Error:" in result.output
