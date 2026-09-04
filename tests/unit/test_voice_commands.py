from legacy_music.commands.voice import _console_safe_text


def test_console_safe_text_replaces_unsupported_progress_glyphs() -> None:
    assert _console_safe_text("  1%|▏", "cp1252") == "  1%|?"


def test_console_safe_text_preserves_unicode_when_supported() -> None:
    assert _console_safe_text("  1%|▏", "utf-8") == "  1%|▏"
