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


def test_redact_provider_shapes_claude_read_probe():
    """Claude READ probe 2026-07-14: these leaked before table alignment."""
    samples = {
        "aws-akid": "AKIAIOSFODNN7EXAMPLE",
        "github": "ghp_abcdefghijklmnopqrstuvwxyz012345",
        "stripe": "sk_live_51AbCdEfGhIjKlMnOpQrStUv",
        "google": "AIzaSyA-abcdefghijklmnopqrstuvwxyz0123",
        "slack": "xoxb-1234567890-abcdefghij",
        "npm": "npm_" + ("a" * 36),
        "sendgrid": "SG." + ("a" * 22) + "." + ("b" * 22),
    }
    for kind, secret in samples.items():
        # Innocent host file body — path-allowed, must still scrub
        body = f"notes.txt says token is {secret} ok"
        out = redact(body)
        assert secret not in out, f"{kind} leaked through redact()"
        assert "REDACTED" in out, f"{kind} not marked redacted"
        # No nested re-redaction (helix colon-format bug)
        assert out.count("[REDACTED:") == 1, f"{kind} nested: {out}"
        # Idempotent
        assert redact(out) == out


def test_redact_preserves_non_secrets():
    s = "use AKIA only as a word fragment AKxx and readme.md"
    # Short / non-matching shapes must pass through
    assert "readme.md" in redact(s)


def test_redact_secret_assign_and_pem():
    assert "supersecretvalue99" not in redact("DB_PASSWORD=supersecretvalue99")
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF6PZGBwODA6Au6rIRI/LRm8C5Mwr\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact(pem)
    assert "MIIEowIBAAKCAQEA" not in out
    assert "REDACTED" in out


def test_deny_message():
    assert "BLOCKED" in deny_read_message(".env")
