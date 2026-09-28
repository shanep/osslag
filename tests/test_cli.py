from datetime import datetime, timezone

import pytest
import typer

from osslag.cli import parse_since_date


def test_parse_since_date_none_means_full_history():
    assert parse_since_date(None) is None


def test_parse_since_date_is_midnight_utc():
    assert parse_since_date("2021-07-31") == datetime(2021, 7, 31, tzinfo=timezone.utc)


def test_parse_since_date_rejects_other_formats():
    with pytest.raises(typer.BadParameter):
        parse_since_date("07/31/2021")
