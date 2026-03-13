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
    assert "0.5.0" in result.output
