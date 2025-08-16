from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import typer
from mistralai import Mistral
from mistralai.models import DocumentURLChunk, ImageURLChunk, TextChunk
from dotenv import load_dotenv
load_dotenv()

app = typer.Typer(add_completion=False, help="UNI OCR CLI")


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


def _as_data_url(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path.name)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    if mime and mime.startswith("image/"):
        return "image", f"data:{mime};base64,{b64}"
    return "document", f"data:{mime or 'application/octet-stream'};base64,{b64}"


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _save_page_images(page: dict, outdir: Path) -> List[str]:
    names: List[str] = []
    for i, img in enumerate(page.get("images", []) or []):
        data_url = img.get("image_base64") or ""
        if ";base64," not in data_url:
            continue
        header, b64 = data_url.split(";base64,", 1)
        ext = ".png"
        if header.startswith("data:image/"):
            subtype = header.split("/")[1].split(";")[0]
            ext = ".jpg" if subtype == "jpeg" else f".{subtype}"
        name = f"page-{page['index']:03d}-img-{i:02d}{ext}"
        (outdir / name).write_bytes(base64.b64decode(b64))
        names.append(name)
    return names


def _pages_markdown(pages: List[dict], outdir: Path) -> str:
    parts: List[str] = ["# OCR Output"]
    for p in pages:
        md = p.get("markdown") or ""
        local = _save_page_images(p, outdir)
        for i, fname in enumerate(local):
            md = md.replace(f"](img-{i}.", f"]({fname}", 1)
        parts.append(f"\n\n<!-- Page {p['index']} -->\n\n{md}")
    return "\n".join(parts)


def _document_chunk_from_input(inp: str):
    if _is_url(inp):
        ext = Path(urlparse(inp).path).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".avif"):
            return ImageURLChunk(image_url=inp)
        return DocumentURLChunk(document_url=inp)
    p = Path(inp).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Not found: {p}")
    kind, data_url = _as_data_url(p)
    return ImageURLChunk(image_url=data_url) if kind == "image" else DocumentURLChunk(document_url=data_url)


def _client_or_die(api_key: Optional[str]) -> Mistral:
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("Missing MISTRAL_API_KEY (set env or use --api-key).")
    return Mistral(api_key=key)


@app.command(help="Run OCR and save Markdown + raw JSON.")
def ocr(
    input: str = typer.Argument(..., help="Path or URL to PDF/image"),
    out: Path = typer.Option(Path("ocr_out"), "--out", help="Output folder"),
    model: str = typer.Option("mistral-ocr-latest", "--model"),
    include_images: bool = typer.Option(True, "--include-images/--no-images", help="Include base64 images in JSON"),
    pages: Optional[str] = typer.Option(None, "--pages", help="Comma separated zero-based indexes, e.g. 0,1,2"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="MISTRAL_API_KEY"),
) -> None:
    client = _client_or_die(api_key)
    chunk = _document_chunk_from_input(input)

    pg_list: Optional[List[int]] = None
    if pages:
        try:
            pg_list = [int(x.strip()) for x in pages.split(",") if x.strip()]
        except ValueError as e:
            raise typer.BadParameter("Invalid --pages. Use comma-separated integers.") from e

    kwargs = dict(model=model, document=chunk, include_image_base64=include_images)
    if pg_list is not None:
        kwargs["pages"] = pg_list

    resp = client.ocr.process(**kwargs)
    payload = json.loads(resp.model_dump_json())

    _ensure_dir(out)
    (out / "raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md = _pages_markdown(payload.get("pages", []) or [], out)
    (out / "doc.md").write_text(md, encoding="utf-8")

    typer.echo(str(out / "doc.md"))
    typer.echo(str(out / "raw.json"))


@app.command(help="Run OCR then ask a chat model to return strict JSON (optionally with JSON Schema).")
def structured(
    input: str = typer.Argument(..., help="Path or URL to PDF/image"),
    out: Path = typer.Option(Path("ocr_out_structured"), "--out"),
    ocr_model: str = typer.Option("mistral-ocr-latest", "--ocr-model"),
    llm_model: str = typer.Option("pixtral-12b-latest", "--llm-model"),
    with_image: bool = typer.Option(True, "--with-image/--no-image", help="Include the image in the chat message"),
    response_format: str = typer.Option("json_object", "--rf", help="json_object | json_schema"),
    schema: Optional[Path] = typer.Option(None, "--schema", help="Path to a JSON Schema file (used with --rf=json_schema)"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Instruction override"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="MISTRAL_API_KEY"),
) -> None:
    client = _client_or_die(api_key)
    chunk = _document_chunk_from_input(input)

    ocr_resp = client.ocr.process(document=chunk, model=ocr_model, include_image_base64=False)
    if not ocr_resp.pages:
        raise RuntimeError("OCR returned no pages.")

    md = ocr_resp.pages[0].markdown
    content = []
    if with_image and isinstance(chunk, ImageURLChunk):
        content.append(chunk)
    instr = prompt or (
        "You are a document verification and extraction AI.\n"
        "\n"
        "SCOPE:\n"
        "- Only accept UNIVERSALLY RECOGNIZED HUMAN IDENTIFICATION documents: passports, national ID cards, Aadhaar, PAN, voter ID, driver's licences, residence permits, student admit cards/hall tickets, bank passbooks used as ID, and similar government/issuer-backed identity proofs.\n"
        "- If the input is NOT an ID document (e.g., bedroom photo, selfie, landscape, invoice, receipt, menu, brochure, flyer, resume/CV, contract, ticket/boarding pass, notes, forms without identity function), return strictly this JSON: {\"status\":\"INVALID\",\"reason\":\"<short reason>\"}.\n"
        "- If you are uncertain, return INVALID.\n"
        "\n"
        "EVIDENCE HEURISTICS (non-exhaustive signals an ID is present):\n"
        "- Phrases/fields like: Government of…, Republic of…, Passport, Aadhaar, PAN, Voter ID, Driver Licence, National ID, Date of Birth, Sex/Gender, Blood Group, Address, ID/UID/Number, Signature, Issuing Authority, Validity/Expiry, Roll/Enrollment (for admit cards), MRZ lines (<<<<<), QR/barcodes tied to identity.\n"
        "- Card/passport layout, official seals/stamps, authority logos, portrait photograph box, machine-readable zones.\n"
        "\n"
        "OUTPUT FORMAT (JSON ONLY, no commentary):\n"
        "- For valid ID docs:\n"
        "  {\n"
        "    \"status\": \"VALID_ID\",\n"
        "    \"doc_type\": \"<e.g., Passport | Aadhaar | PAN | Voter ID | Driver Licence | Admit Card | Bank Passbook>\",\n"
        "    \"issuer\": \"<issuing authority/org if known>\",\n"
        "    \"country\": \"<if inferable>\",\n"
        "    \"data\": {\n"
        "      \"name\": \"<full name if present>\",\n"
        "      \"dob\": \"<YYYY-MM-DD or as printed>\",\n"
        "      \"id_number\": \"<primary ID number>\",\n"
        "      \"address\": \"<structured or as printed>\",\n"
        "      \"phone\": \"<if present>\",\n"
        "      \"issue_date\": \"<if present>\",\n"
        "      \"expiry_date\": \"<if present>\",\n"
        "      \"extra\": {\"<other relevant fields>\": \"<values>\"}\n"
        "    }\n"
        "  }\n"
        "- For non-ID or uncertain cases:\n"
        "  {\"status\":\"INVALID\",\"reason\":\"<short reason>\"}\n"
        "\n"
        "HARD RULES:\n"
        "- Return ONLY JSON as described above.\n"
        "- Never invent content not supported by the document.\n"
        "- If OCR text is insufficient and the image is unavailable or non-document, return INVALID.\n"
    )
    content.append(TextChunk(text=f"{instr}\n\nOCR Markdown:\n\n{md}"))

    if response_format == "json_schema":
        if not schema:
            raise typer.BadParameter("--schema is required when --rf=json_schema")
        rf = {"type": "json_schema", "json_schema": json.loads(schema.read_text())}
    else:
        rf = {"type": "json_object"}

    chat = client.chat.complete(
        model=llm_model,
        messages=[{"role": "user", "content": content}],
        response_format=rf,
        temperature=0,
    )

    raw = chat.choices[0].message.content
    obj = json.loads(raw) if isinstance(raw, str) else raw

    _ensure_dir(out)
    (out / "structured.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
