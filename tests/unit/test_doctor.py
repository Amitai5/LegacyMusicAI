from legacy_music.commands import doctor as doctor_module


def test_collect_tool_checks_marks_discovered_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda name: f"/tools/{name}" if name in {"git", "uv"} else None,
    )

    checks = doctor_module.collect_tool_checks()

    by_name = {check.name: check for check in checks}
    assert by_name["git"].is_available is True
    assert by_name["git"].required_for_scaffold is True
    assert by_name["ffmpeg"].is_available is False
    assert by_name["ffmpeg"].required_for_scaffold is False
