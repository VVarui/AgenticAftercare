"""Note generator guards. No GPU, no model download. Run:  pytest tests/test_note_generator.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import note_generator_aftercare as ng  # noqa: E402


def test_clean_note_normal_text_has_no_flags():
    text, flags = ng.clean_note("The patient has had knee pain for two days. She is taking ibuprofen twice a day.")
    assert flags == [] and text.startswith("The patient")


def test_clean_note_cuts_continued_dialogue():
    text, flags = ng.clean_note("Knee pain for two days and mild swelling today.\nDoctor: How are you?")
    assert "continued_dialogue" in flags and "Doctor:" not in text


def test_clean_note_removes_repeated_sentences():
    s = "The patient is able to walk around the room without any difficulty."
    text, flags = ng.clean_note(f"{s} {s} {s}")
    assert "repetition" in flags and text.count("walk around the room") == 1


def test_clean_note_flags_truncated_and_very_short():
    _, flags = ng.clean_note("The patient reports knee pain that started two days ago and then")
    assert "truncated" in flags
    _, flags = ng.clean_note("Lisinopril.")
    assert "very_short" in flags


def test_prompt_format_matches_training():
    p = ng.build_prompt("Doctor: hi\nPatient: knee hurts", "section")
    assert p.startswith("[INST] Summarize the following patient conversation")
    assert p.endswith("\n\nNOTE: [/INST]")


def test_format_dialogue_maps_roles():
    d = ng.format_dialogue([{"role": "patient", "content": "knee hurts"}, {"role": "assistant", "content": "since when?"}])
    assert d == "Patient: knee hurts\nDoctor: since when?"


def test_disabled_without_adapter_path():
    g = ng.NoteGenerator(adapter_path="")
    assert not g.enabled and "NOTE_ADAPTER_PATH" in g.disabled_reason()


def test_disabled_when_adapter_folder_missing(tmp_path):
    g = ng.NoteGenerator(adapter_path=str(tmp_path / "nope"))
    assert "not found" in g.disabled_reason()


def test_disabled_without_gpu_even_if_adapter_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(ng, "ALLOW_CPU", False)
    monkeypatch.setattr(ng.NoteGenerator, "_gpu_available", staticmethod(lambda: False))
    g = ng.NoteGenerator(adapter_path=str(tmp_path))
    assert not g.enabled and "GPU" in g.disabled_reason()
    with pytest.raises(RuntimeError):
        g.generate_with_flags([{"role": "patient", "content": "hi"}])


def test_enabled_with_gpu_and_model_version(tmp_path, monkeypatch):
    monkeypatch.setattr(ng.NoteGenerator, "_gpu_available", staticmethod(lambda: True))
    folder = tmp_path / "llama-2-aftercare-v4"
    folder.mkdir()
    g = ng.NoteGenerator(adapter_path=str(folder))
    assert g.enabled and g.model_version == "llama-2-aftercare-v4"


def test_demo_mode_returns_labelled_recorded_note(monkeypatch):
    import importlib
    monkeypatch.setenv("NOTE_DEMO_MODE", "1")
    monkeypatch.delenv("NOTE_ADAPTER_PATH", raising=False)
    demo = importlib.reload(ng)
    try:
        gen = demo.NoteGenerator(adapter_path="")
        assert gen.enabled and gen.demo_mode and gen.disabled_reason() is None
        assert gen.model_version.startswith("DEMO")
        text, flags = gen.generate_with_flags([{"role": "patient", "content": "My knee incision looks red"}])
        assert "knee replacement" in text and flags == []
        text, _ = gen.generate_with_flags([{"role": "patient", "content": "I have a fever and a cough"}])
        assert text.startswith("The patient presents today with a fever")
        text, _ = gen.generate_with_flags([{"role": "patient", "content": "short of breath on the stairs"}])
        assert "furosemide" in text
        with pytest.raises(ValueError):
            gen.generate_with_flags([])
    finally:
        monkeypatch.delenv("NOTE_DEMO_MODE", raising=False)
        importlib.reload(ng)
    assert not ng.NoteGenerator(adapter_path="").enabled
