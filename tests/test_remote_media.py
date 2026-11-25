import asyncio
import base64
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import AudioContent, EmbeddedResource, ImageContent, TextContent

from telegram_mcp import remote_media


def _summary(result):
    assert isinstance(result[0], TextContent)
    return json.loads(result[0].text)


@pytest.mark.asyncio
async def test_small_document_is_embedded_as_blob(tmp_path):
    doc = tmp_path / "plan.dxf"
    doc.write_bytes(b"0\nSECTION\n")
    result = await remote_media.build_download_result(
        path=doc, kind="document", filename="plan.dxf", uri="telegram://media/1/2", cached=False
    )
    summary = _summary(result)
    assert summary["Embedded"] is True
    assert summary["Filename"] == "plan.dxf"
    assert isinstance(result[1], EmbeddedResource)
    assert str(result[1].resource.uri) == "telegram://media/1/2"
    assert base64.b64decode(result[1].resource.blob) == b"0\nSECTION\n"


@pytest.mark.asyncio
async def test_image_and_audio_use_native_content_blocks(tmp_path):
    from PIL import Image

    img = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(img)
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS")

    image_result = await remote_media.build_download_result(
        path=img, kind="image", filename="photo.png", uri="telegram://media/1/3", cached=False
    )
    audio_result = await remote_media.build_download_result(
        path=audio, kind="audio", filename="voice.ogg", uri="telegram://media/1/4", cached=False
    )
    assert isinstance(image_result[1], ImageContent)
    assert image_result[1].mimeType == "image/png"
    assert isinstance(audio_result[1], AudioContent)


@pytest.mark.asyncio
async def test_large_image_is_downscaled_and_original_kept(tmp_path, monkeypatch):
    from PIL import Image

    img = tmp_path / "big.png"
    Image.effect_noise((600, 600), 60).convert("RGB").save(img)
    original = img.read_bytes()
    cap = len(original) // 2
    monkeypatch.setattr(remote_media, "INLINE_MAX_BYTES", cap)

    result = await remote_media.build_download_result(
        path=img, kind="image", filename="big.png", uri="telegram://media/1/5", cached=False
    )
    assert _summary(result)["Embedded"] is True
    assert result[1].mimeType == "image/jpeg"
    assert len(base64.b64decode(result[1].data)) <= cap
    assert "downscaled JPEG copy" in result[-1].text
    assert img.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("gs") is None, reason="Ghostscript not installed")
async def test_large_pdf_is_recompressed_with_ghostscript(tmp_path, monkeypatch):
    from PIL import Image

    pdf = tmp_path / "scan.pdf"
    pages = [Image.effect_noise((1200, 1600), 60).convert("RGB") for _ in range(2)]
    pages[0].save(pdf, save_all=True, append_images=pages[1:], quality=95, resolution=300)
    monkeypatch.setattr(remote_media, "INLINE_MAX_BYTES", pdf.stat().st_size // 2)

    result = await remote_media.build_download_result(
        path=pdf, kind="document", filename="scan.pdf", uri="telegram://media/1/6", cached=False
    )
    assert _summary(result)["Embedded"] is True
    blob = base64.b64decode(result[1].resource.blob)
    assert blob.startswith(b"%PDF-")
    assert len(blob) <= remote_media.INLINE_MAX_BYTES
    assert "recompressed copy" in result[-1].text


@pytest.mark.asyncio
async def test_oversized_file_reports_path_and_url_without_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_media, "INLINE_MAX_BYTES", 4)
    blob = tmp_path / "archive.zip"
    blob.write_bytes(b"PK" + b"x" * 100)
    result = await remote_media.build_download_result(
        path=blob,
        kind="document",
        filename="archive.zip",
        uri="telegram://media/1/7",
        cached=False,
        download_url="https://example.test/media/archive.zip?sig=1",
    )
    summary = _summary(result)
    assert summary["Embedded"] is False
    assert summary["DownloadUrl"].startswith("https://example.test/")
    assert len(result) == 2 and "DownloadUrl" in result[1].text


@pytest.mark.asyncio
async def test_pdf_over_compress_input_limit_is_not_sent_to_ghostscript(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_media, "INLINE_MAX_BYTES", 4)
    monkeypatch.setattr(remote_media, "PDF_MAX_COMPRESS_INPUT", 10)

    async def no_gs(*args, **kwargs):
        raise AssertionError("Ghostscript must not run")

    monkeypatch.setattr(remote_media, "_gs_compress", no_gs)
    pdf = tmp_path / "huge.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 100)
    result = await remote_media.build_download_result(
        path=pdf, kind="document", filename="huge.pdf", uri="telegram://media/1/8", cached=False
    )
    assert _summary(result)["Embedded"] is False


@pytest.mark.asyncio
async def test_slow_job_reports_in_progress_then_returns_result(monkeypatch):
    monkeypatch.setattr(remote_media, "JOB_WAIT_SECONDS", 0.05)
    release = asyncio.Event()
    runs = []

    async def job():
        runs.append(1)
        await release.wait()
        return ["done"]

    first = await remote_media.run_job("k", job)
    assert json.loads(first[0].text)["InProgress"] is True
    release.set()
    second = await remote_media.run_job("k", job)
    assert second == ["done"]
    assert runs == [1]  # the second call joined the running job


def test_safe_original_name_strips_paths_and_switches():
    assert remote_media.safe_original_name("../../etc/passwd", "f") == "passwd"
    assert remote_media.safe_original_name("-rf.pdf", "f") == "rf.pdf"
    assert remote_media.safe_original_name("a<b>:c?.pdf", "f") == "a_b_c_.pdf"
    assert remote_media.safe_original_name(None, "fallback") == "fallback"


def test_cad_and_vector_image_mime_types_are_documents():
    def msg(mime):
        return SimpleNamespace(file=SimpleNamespace(mime_type=mime), photo=None)

    assert remote_media.media_kind(msg("image/vnd.dxf")) == "document"
    assert remote_media.media_kind(msg("image/svg+xml")) == "document"
    assert remote_media.media_kind(msg("image/jpeg")) == "image"
    assert remote_media.media_kind(msg("audio/ogg")) == "audio"
    assert remote_media.media_kind(msg("video/mp4")) == "video"
    assert remote_media.media_kind(msg("application/pdf")) == "document"
