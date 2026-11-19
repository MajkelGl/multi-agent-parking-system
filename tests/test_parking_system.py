import pytest
from unittest.mock import patch, MagicMock
from src.bot.guardrails import guardrails
from src.bot.admin_app import check_spot_availability, fetch_pending_reservation, update_reservation_status, send_to_fastapi

def test_guardrails_tokenizes_sensitive_data():
    """Ensure PII (names, phone numbers, license plates) are successfully tokenized and stored in the vault."""
    guardrails.clear_vault()
    text = "My name is Jan Kowalski, phone +48500600700, plate DW12345 in sector A."
    
    tokenized = guardrails.tokenize(text)
    
    assert "Jan Kowalski" not in tokenized
    assert "+48500600700" not in tokenized
    assert "DW12345" not in tokenized
    assert "sector A" in tokenized or "Sector A" in tokenized
    
    assert len(guardrails.vault) >= 3
    
    detokenized = guardrails.detokenize(tokenized)
    assert detokenized == text
    guardrails.clear_vault()

def test_guardrails_ignores_conversational_words_and_sectors():
    """Ensure common words, conversational filler, and parking sectors are not falsely flagged as PII."""
    guardrails.clear_vault()
    text = "Hi, I want to check the price for parking in sector B or VIP."
    
    tokenized = guardrails.tokenize(text)
    
    assert len(guardrails.vault) == 0
    assert tokenized == text
    guardrails.clear_vault()

@patch("src.bot.admin_app.psycopg2.connect")
def test_check_spot_availability_no_conflict(mock_connect):
    """Ensure check_spot_availability returns True when no overlapping approved reservation exists."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    is_available = check_spot_availability(
        spot_id=1,
        start_time="2026-11-20 08:00:00",
        end_time="2026-11-20 17:00:00",
        current_res_id=5
    )

    assert is_available is True
    mock_cursor.execute.assert_called_once()


@patch("src.bot.admin_app.psycopg2.connect")
def test_check_spot_availability_with_conflict(mock_connect):
    """Ensure check_spot_availability returns False when an overlapping approved reservation exists."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (10,)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    is_available = check_spot_availability(
        spot_id=1,
        start_time="2026-11-20 08:00:00",
        end_time="2026-11-20 17:00:00",
        current_res_id=5
    )

    assert is_available is False

@patch("src.bot.admin_app.psycopg2.connect")
def test_fetch_pending_reservation_found(mock_connect):
    """Ensure fetch_pending_reservation correctly parses database rows into dictionary format."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (
        1, 2, "Anna Nowak", "+48600700800", "WA99999", 
        "2026-11-21 09:00:00", "2026-11-21 17:00:00", "PENDING"
    )
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    res = fetch_pending_reservation()

    assert res is not None
    assert res["id"] == 1
    assert res["full_name"] == "Anna Nowak"
    assert res["status"] == "PENDING"
    assert res["license_plate"] == "WA99999"


@patch("src.bot.admin_app.requests.post")
def test_send_to_fastapi_success(mock_post):
    """Ensure send_to_fastapi returns True upon successful HTTP 200 response from FastAPI server."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    sample_data = {
        "full_name": "Test User",
        "license_plate": "KR12345",
        "start_time": "2026-11-22 10:00:00",
        "end_time": "2026-11-22 12:00:00"
    }

    success = send_to_fastapi(sample_data)
    assert success is True
    mock_post.assert_called_once()


@patch("src.bot.admin_app.requests.post")
def test_send_to_fastapi_failure(mock_post):
    """Ensure send_to_fastapi returns False and handles connection errors gracefully."""
    mock_post.side_effect = Exception("Connection refused")

    sample_data = {
        "full_name": "Test User",
        "license_plate": "KR12345",
        "start_time": "2026-11-22 10:00:00",
        "end_time": "2026-11-22 12:00:00"
    }

    success = send_to_fastapi(sample_data)
    assert success is False