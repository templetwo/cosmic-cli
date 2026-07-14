from cosmic_cli.secrets import deny_read_message, is_sensitive_path, redact


def test_sensitive_paths():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("foo/.env.local")
    assert is_sensitive_path("id_rsa")
    assert is_sensitive_path("cert.pem")
    assert is_sensitive_path(".ssh/config")
    assert not is_sensitive_path("readme.md")
    assert not is_sensitive_path("app.py")


def test_redact_key_shapes():
    s = "XAI_API_KEY=xai-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    out = redact(s)
    assert "xai-ABCDEF" not in out
    assert "REDACTED" in out
    assert "REDACTED" in redact(
        "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz"
    )


def test_deny_message():
    assert "BLOCKED" in deny_read_message(".env")
