#!/usr/bin/env python3
"""Generate a PPTX for thesis using a PDF and a Jupyter notebook.

Usage:
  python "scripts/generate_thesis_pptx.py" --pdf thesis.pdf --notebook Notebook.ipynb --output thesis.pptx

This script extracts text from the first pages of the PDF, converts markdown cells
to slides, includes code cells as monospaced text slides, and embeds images found
in notebook outputs (image/png, image/jpeg).
"""
import argparse
import base64
import os
import textwrap
import uuid
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    import nbformat
    import PyPDF2
except Exception as e:
    raise SystemExit("Missing dependencies. Install from requirements.txt: pip install -r requirements.txt\n" + str(e))


def extract_pdf_text(pdf_path, max_pages=50):
    text = []
    reader = PyPDF2.PdfReader(str(pdf_path))
    pages = min(max_pages, len(reader.pages))
    for i in range(pages):
        try:
            page = reader.pages[i]
            page_text = page.extract_text() or ""
            text.append(page_text)
        except Exception:
            continue
    return "\n\n".join(text).strip()


def save_image_from_base64(b64, out_dir, ext="png"):
    data = b64
    if isinstance(data, (list, tuple)):
        data = "".join(data)
    # Data may already be bytes or a base64 string
    if isinstance(data, str):
        data = data.encode('utf-8')
    try:
        raw = base64.b64decode(data)
    except Exception:
        # maybe already bytes
        raw = data
    fname = out_dir / f"img_{uuid.uuid4().hex}.{ext}"
    with open(fname, 'wb') as f:
        f.write(raw)
    return fname


def extract_notebook_cells(nb_path, images_outdir):
    nb = nbformat.read(str(nb_path), as_version=4)
    cells = []
    images_outdir = Path(images_outdir)
    images_outdir.mkdir(parents=True, exist_ok=True)
    for cell in nb.cells:
        entry = {"cell_type": cell.get("cell_type"), "source": cell.get("source", "")}
        imgs = []
        if cell.get("outputs"):
            for out in cell.get("outputs", []):
                data = out.get("data") or {}
                for mime in ("image/png", "image/jpeg"):
                    if mime in data:
                        ext = "png" if mime == "image/png" else "jpg"
                        img_path = save_image_from_base64(data[mime], images_outdir, ext=ext)
                        imgs.append(str(img_path))
        entry["images"] = imgs
        cells.append(entry)
    return cells


def chunk_text(text, size=800):
    # split by paragraphs then into chunks
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    curr_len = 0
    for p in paragraphs:
        if curr_len + len(p) + 2 <= size:
            current.append(p)
            curr_len += len(p) + 2
        else:
            if current:
                chunks.append('\n\n'.join(current))
            current = [p]
            curr_len = len(p) + 2
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


def make_pptx(output_path, title, pdf_summary, nb_cells, images_dir):
    prs = Presentation()
    # Title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = title or "Thesis Presentation"
    try:
        slide.placeholders[1].text = "Generated with script"
    except Exception:
        pass

    # PDF summary slides
    if pdf_summary:
        chunks = chunk_text(pdf_summary, size=1000)
        for i, c in enumerate(chunks):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            shapes = slide.shapes
            title_box = shapes.title if shapes.title else None
            if title_box is not None:
                title_box.text = f"PDF Summary (part {i+1})"
            left = Inches(0.5)
            top = Inches(1.2)
            width = Inches(9)
            height = Inches(5)
            txBox = slide.shapes.add_textbox(left, top, width, height)
            tf = txBox.text_frame
            tf.word_wrap = True
            for para in c.split('\n'):
                p = tf.add_paragraph()
                p.text = para
                p.font.size = Pt(14)

    # Notebook cells
    for idx, cell in enumerate(nb_cells, start=1):
        ctype = cell.get("cell_type")
        src = cell.get("source", "")
        imgs = cell.get("images", [])
        if ctype == "markdown":
            chunks = chunk_text(src, size=1000)
            for i, c in enumerate(chunks):
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                try:
                    slide.shapes.title.text = f"Notebook: Markdown (cell {idx})"
                except Exception:
                    pass
                left = Inches(0.5)
                top = Inches(1.2)
                width = Inches(9)
                height = Inches(5)
                txBox = slide.shapes.add_textbox(left, top, width, height)
                tf = txBox.text_frame
                tf.word_wrap = True
                for para in c.split('\n'):
                    p = tf.add_paragraph()
                    p.text = para
                    p.font.size = Pt(14)
        elif ctype == "code":
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            try:
                slide.shapes.title.text = f"Notebook: Code (cell {idx})"
            except Exception:
                pass
            left = Inches(0.5)
            top = Inches(1.2)
            width = Inches(9)
            height = Inches(5)
            txBox = slide.shapes.add_textbox(left, top, width, height)
            tf = txBox.text_frame
            tf.word_wrap = True
            code_lines = src.split('\n')
            for ln in code_lines:
                p = tf.add_paragraph()
                p.text = ln
                p.font.size = Pt(10)
        # images from outputs
        for img in imgs:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            left = Inches(1)
            top = Inches(1)
            try:
                slide.shapes.add_picture(str(img), left, top, width=Inches(8))
            except Exception:
                continue

    prs.save(str(output_path))


def main():
    p = argparse.ArgumentParser(description="Generate PPTX from PDF and Jupyter notebook")
    p.add_argument("--pdf", required=True, help="Path to PDF file to summarize")
    p.add_argument("--notebook", required=True, help="Path to .ipynb file")
    p.add_argument("--output", required=True, help="Output .pptx path")
    p.add_argument("--title", default="Thesis Presentation", help="Presentation title")
    p.add_argument("--pdf-pages", type=int, default=2, help="Pages of PDF to extract for summary")
    args = p.parse_args()

    pdf_path = Path(args.pdf)
    nb_path = Path(args.notebook)
    out_path = Path(args.output)
    tmp_images = Path('.pptx_tmp_images')
    tmp_images.mkdir(parents=True, exist_ok=True)

    print(f"Extracting first {args.pdf_pages} pages from {pdf_path}...")
    pdf_text = extract_pdf_text(pdf_path, max_pages=args.pdf_pages)

    print(f"Reading notebook {nb_path} and extracting images to {tmp_images}...")
    nb_cells = extract_notebook_cells(nb_path, tmp_images)

    print(f"Creating PPTX {out_path}...")
    make_pptx(out_path, args.title, pdf_text, nb_cells, tmp_images)
    print("Done. PPTX saved to", out_path)


if __name__ == '__main__':
    main()
