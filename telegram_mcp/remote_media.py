"""Remote-friendly results for download_media.

The first content block is a JSON summary with the path on the server (and
the signed TELEGRAM_MEDIA_BASE_URL link when configured). For files up to
``INLINE_MAX_BYTES`` the bytes are also embedded in the tool result, so MCP
clients on other machines can use them without access to this host's
filesystem or to the media URL:

* images  -> ImageContent (downscaled JPEG copy when the original is too big)
* audio   -> AudioContent
* documents -> EmbeddedResource blob (PDFs over the cap are recompressed with
  Ghostscript, /ebook then /screen, and the copy is embedded when it fits)
* videos  -> never embedded

Compressed copies live next to the original as hidden
``.<name>.mcp-compressed.<ext>`` files and are reused while newer than the
original, so repeated calls are cheap. The original file is never modified.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

INLINE_MAX_BYTES = int(os.getenv("TELEGRAM_INLINE_MEDIA_MAX_BYTES", str(5 * 1024 * 1024)))
PDF_COMPRESS_TIMEOUT = float(os.getenv("TELEGRAM_PDF_COMPRESS_TIMEOUT", "300"))
PDF_MAX_COMPRESS_INPUT = int(os.getenv("TELEGRAM_PDF_MAX_COMPRESS_INPUT", str(60 * 1000 * 1000)))
_PDF_PRESETS = ("/ebook", "/screen")
_IMAGE_STEPS = ((2560, 85), (2048, 80), (1600, 75), (1280, 70), (1024, 65))

_UNSAFE_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


def safe_original_name(name: Optional[str], fallback: str) -> str:
    """Sanitise a sender-provided filename so it is a single safe path part."""
    if not name:
        return fallback
    name = _UNSAFE_CHARS.sub("_", Path(name).name).strip(" .")
    name = name.lstrip("-")  # never looks like a CLI switch
    if not name:
        return fallback
    stem, ext = os.path.splitext(name)
    return stem[:120] + ext[:16]


# Raster formats MCP clients can render as ImageContent as-is.
_RENDERABLE_IMAGES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
# Raster formats Pillow can convert to an embeddable JPEG copy.
_CONVERTIBLE_IMAGES = {"image/bmp", "image/x-ms-bmp", "image/tiff"}


def media_kind(message: Any) -> str:
    """Classify a Telethon message/document as image, audio, video or document.

    Only raster images a client can display count as "image"; vector or CAD
    formats Telegram labels image/* (DXF, SVG, ...) are documents.
    """
    f = getattr(message, "file", None)
    mime = (getattr(f, "mime_type", None) or "").lower()
    if (
        getattr(message, "photo", None) is not None
        or mime in _RENDERABLE_IMAGES | _CONVERTIBLE_IMAGES
    ):
        return "image"
    if (
        getattr(message, "voice", None)
        or getattr(message, "audio", None)
        or mime.startswith("audio/")
    ):
        return "audio"
    if (
        getattr(message, "video", None)
        or getattr(message, "gif", None)
        or mime.startswith("video/")
    ):
        return "video"
    return "document"


def original_filename(message: Any) -> Optional[str]:
    f = getattr(message, "file", None)
    return getattr(f, "name", None) if f is not None else None


def _guess_mime(path: Path, kind: str) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    return {"image": "image/jpeg", "audio": "audio/ogg", "video": "video/mp4"}.get(
        kind, "application/octet-stream"
    )


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"%PDF-" in fh.read(1024)
    except OSError:
        return False


def _compressed_path(path: Path, ext: str) -> Path:
    return path.with_name(f".{path.name}.mcp-compressed{ext}")


def _fresh(copy: Path, original: Path) -> bool:
    try:
        return copy.stat().st_mtime >= original.stat().st_mtime
    except OSError:
        return False


async def _gs_compress(src: Path, dst: Path, preset: str) -> bool:
    gs = shutil.which("gs")
    if not gs:
        return False
    fd, tmp = tempfile.mkstemp(prefix=".tg-mcp-pdf-", suffix=".pdf", dir=str(dst.parent))
    os.close(fd)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path(tmp).parent)}
    try:
        proc = await asyncio.create_subprocess_exec(
            gs,
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=pdfwrite",
            f"-dPDFSETTINGS={preset}",
            f"-sOutputFile={tmp}",
            # "./" guard: a filename starting with "-" is never a gs switch
            str(src) if src.is_absolute() else f"./{src}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=PDF_COMPRESS_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        if rc != 0 or not _is_pdf(Path(tmp)):
            return False
        os.replace(tmp, dst)
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


async def _compressed_pdf(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return (copy, preset) of a recompressed PDF that fits the cap, or (None, None).

    Ghostscript on a large, mostly-vector PDF is slow on a Raspberry Pi
    (a 503-page, 81 MB project PDF took ~4.5 min per preset and still came out
    at 40 MB / 32 MB), so: inputs over PDF_MAX_COMPRESS_INPUT are skipped,
    /screen is only tried when /ebook got reasonably close, oversized copies
    are deleted and a ``.mcp-nocompress`` marker avoids retrying on every call.
    """
    marker = path.with_name(f".{path.name}.mcp-nocompress")
    if _fresh(marker, path) or path.stat().st_size > PDF_MAX_COMPRESS_INPUT:
        return None, None
    for preset in _PDF_PRESETS:
        out = path.with_name(f".{path.name}.mcp-compressed{preset.strip('/')}.pdf")
        if not _fresh(out, path):
            if not await _gs_compress(path, out, preset):
                continue
        size = out.stat().st_size
        if size <= INLINE_MAX_BYTES:
            return out, preset
        out.unlink(missing_ok=True)
        if size > 2 * INLINE_MAX_BYTES:
            break  # the next, lossier preset will not get under the cap either
    marker.touch()
    return None, None


def _downscaled_image(path: Path) -> Optional[Path]:
    """Return a JPEG copy small enough to embed, or None."""
    out = _compressed_path(path, ".jpg")
    if _fresh(out, path) and out.stat().st_size <= INLINE_MAX_BYTES:
        return out
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            for max_side, quality in _IMAGE_STEPS:
                copy = im.copy()
                copy.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                copy.save(buf, "JPEG", quality=quality, optimize=True)
                if buf.tell() <= INLINE_MAX_BYTES:
                    tmp = out.with_suffix(".tmp")
                    tmp.write_bytes(buf.getvalue())
                    os.replace(tmp, out)
                    return out
    except Exception:
        return None
    return None


JOB_WAIT_SECONDS = float(os.getenv("TELEGRAM_DOWNLOAD_WAIT_SECONDS", "45"))
_JOBS: dict = {}


def _forget(key: str, task: "asyncio.Task") -> None:
    if _JOBS.get(key) is task:
        _JOBS.pop(key, None)
    if not task.cancelled():
        task.exception()  # mark retrieved; errors were already reported to a waiter or are retried


async def run_job(key: str, factory) -> List[Any]:
    """Run a download/compress job in the background and wait for it briefly.

    The job is shielded from the caller's cancellation (client or server
    timeouts), so it keeps running and fills the on-disk cache; a later call
    with the same arguments either joins the running job or hits the cache.
    """
    task = _JOBS.get(key)
    if task is None or task.done():
        task = asyncio.create_task(factory())
        _JOBS[key] = task
        task.add_done_callback(lambda t, k=key: _forget(k, t))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=JOB_WAIT_SECONDS)
    except asyncio.TimeoutError:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "Success": False,
                        "InProgress": True,
                        "Message": (
                            "The file is still being downloaded and/or compressed in the "
                            "background on the server. Call download_media again with the "
                            "same arguments in about a minute; the finished result is "
                            "cached, so the next call returns at once."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


async def build_download_result(
    *,
    path: Path,
    kind: str,
    filename: str,
    uri: str,
    cached: bool,
    download_url: Optional[str] = None,
) -> List[Any]:
    """Build the content list returned by download_media."""
    size = path.stat().st_size
    message = f"Media downloaded to {path}." + (" (cached)" if cached else "")
    if download_url:
        message += f" From another machine, fetch it from {download_url} (expires in an hour)."
    summary = {
        "Success": True,
        "Message": message,
        "MediaType": kind,
        "Filename": filename,
        "Path": str(path),
        "Size": size,
        "Embedded": False,
    }
    if download_url:
        summary["DownloadUrl"] = download_url
    blocks: List[Any] = []
    note: Optional[str] = None
    mime = _guess_mime(path, kind)

    if kind == "image":
        src = path
        if mime not in _RENDERABLE_IMAGES and size <= INLINE_MAX_BYTES:
            # e.g. BMP/TIFF: embed a JPEG copy the client can render
            src = _downscaled_image(path)
            mime = "image/jpeg"
        elif size > INLINE_MAX_BYTES:
            src = _downscaled_image(path)
            if src is not None:
                note = (
                    f"The original image is {size / 1e6:.1f} MB, over the "
                    f"{INLINE_MAX_BYTES >> 20} MiB embedding cap; the embedded image is a "
                    f"downscaled JPEG copy ({src.stat().st_size / 1e6:.1f} MB). "
                    "The original is kept at Path."
                )
                mime = "image/jpeg"
        if src is not None and src.stat().st_size <= INLINE_MAX_BYTES:
            blocks.append(ImageContent(type="image", data=_b64(src), mimeType=mime))

    elif kind == "audio":
        if size <= INLINE_MAX_BYTES:
            blocks.append(AudioContent(type="audio", data=_b64(path), mimeType=mime))

    elif kind == "document":
        src: Optional[Path] = path
        if size > INLINE_MAX_BYTES:
            src = None
            if _is_pdf(path):
                src, preset = await _compressed_pdf(path)
                if src is not None:
                    note = (
                        f"The original PDF is {size / 1e6:.1f} MB, over the "
                        f"{INLINE_MAX_BYTES >> 20} MiB embedding cap; the embedded document "
                        f"is a recompressed copy (Ghostscript {preset}, "
                        f"{src.stat().st_size / 1e6:.1f} MB). The original is kept at Path."
                    )
        if src is not None:
            blocks.append(
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(uri=uri, mimeType=mime, blob=_b64(src)),
                )
            )

    if blocks:
        summary["Embedded"] = True
    elif kind == "video":
        note = "Videos are not embedded; the file is available at Path on the server."
    elif size > INLINE_MAX_BYTES:
        where = "at DownloadUrl (from another machine) and at Path" if download_url else "at Path"
        note = (
            f"The file is {size / 1e6:.1f} MB, over the {INLINE_MAX_BYTES >> 20} MiB "
            "embedding cap and cannot be compressed enough to embed (large PDFs with many "
            f"pages stay big even after Ghostscript). It is available {where} on the "
            "server; send_file can also forward it inside Telegram from that Path."
        )

    out: List[Any] = [
        TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, indent=2))
    ]
    out.extend(blocks)
    if note:
        out.append(TextContent(type="text", text=note))
    return out
