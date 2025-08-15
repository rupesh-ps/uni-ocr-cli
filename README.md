# UNI OCR CLI

Command-line wrapper around Mistral OCR that:

* Runs OCR on PDFs, images, or URLs (`mistral-ocr-latest`)
* Writes Markdown (with page images) and the raw JSON response
* Optionally produces strict, structured JSON via a follow-up LLM call (e.g., `pixtral-12b-latest`) with either a generic JSON object or validation against your JSON Schema
* Built with Typer and uses `uv` as the package manager

---

## Features

* Inputs: local files or URLs (PDF, PNG/JPG/WebP/TIFF, etc.)
* Outputs:

  * `doc.md` — stitched per-page Markdown; page images saved alongside
  * `raw.json` — OCR API response
  * `structured.json` — LLM-produced structured JSON (optional)
* Controls:

  * Select pages: `--pages 0,2,5`
  * Include/omit base64 images inside OCR JSON
  * Provide a custom prompt or a JSON Schema for strict structured output

---

## Prerequisites

* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv) installed
* Mistral API key in `MISTRAL_API_KEY`

  * macOS/Linux: `export MISTRAL_API_KEY=sk-...`
  * Windows (PowerShell): `$env:MISTRAL_API_KEY="sk-..."`

---

## Quick start (cloned repo)

```bash
git clone <this-repo-url>.git
cd <repo>

# ensure your API key is set 
export MISTRAL_API_KEY=sk-...

# run the CLI (uv will resolve deps automatically)
uv run uni-ocr --help
```

Common one-liner from this repo:

```bash
uv run uni-ocr structured ./samples/adhaar.jpg --with-image --llm-model pixtral-12b-latest > result.json
```

---

## Usage

### 1) OCR → Markdown + raw JSON

Local PDF:

```bash
uv run uni-ocr ocr ./samples/sample.pdf --out ocr_out
```

Local image:

```bash
uv run uni-ocr ocr ./samples/id.jpg --out ocr_img
```

Remote URL (auto-detects image vs document):

```bash
uv run uni-ocr ocr "https://example.com/file.pdf" --out ocr_url
```

Process specific pages (zero-based):

```bash
uv run uni-ocr ocr ./samples/long.pdf --pages 0,3,5
```

Exclude base64 images from the OCR JSON:

```bash
uv run uni-ocr ocr ./samples/id.jpg --no-images
```

**Outputs:**

```
<out>/
├── doc.md
├── raw.json
├── page-000-img-00.jpg
└── page-000-img-01.png
```

### 2) OCR → Structured JSON

Vision model (includes the image in the chat request):

```bash
uv run uni-ocr structured ./samples/adhaar.jpg --with-image --llm-model pixtral-12b-latest > result.json
```

Text-only model (Markdown only):

```bash
uv run uni-ocr structured ./samples/id.jpg --no-image --llm-model ministral-8b-latest > result.json
```

Custom prompt:

```bash
uv run uni-ocr structured ./samples/id.jpg \
  --with-image \
  --prompt "Return JSON with keys: DOC_TYPE, ISSUER, PEOPLE[], DATES[] only."
```

**Output:**

```
ocr_out_structured/
└── structured.json
```

### 3) Strict output with your JSON Schema

Create `schema.json`:

```json
{
  "type": "object",
  "properties": {
    "DOC_TYPE": { "type": "string" },
    "ISSUER":   { "type": "string" },
    "ENTITIES": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["DOC_TYPE"]
}
```

Run:

```bash
uv run uni-ocr structured ./samples/id.jpg \
  --rf json_schema \
  --schema schema.json \
  --with-image \
  --llm-model pixtral-12b-latest > structured.json
```

---

## Tips

* Use `--no-images` if you want a smaller, more diff-friendly `raw.json`.
* For multi-page PDFs, `--pages` helps target just the relevant pages.
* Keep your prompt concise and deterministic; `temperature=0` is used by default.

---
