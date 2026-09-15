"""Step 1:解析所有檔案 → chunks.jsonl + violations.jsonl + failed_files.jsonl

跑這個之前要先:
  1. make unzip(解壓 data/raw/zips → data/raw/)
  2. 確認 LibreOffice、Tesseract 已安裝
"""
import json
import multiprocessing
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

WORKERS = max(1, min(multiprocessing.cpu_count() - 1, 4))  # 最多 4 個 worker，保留 1 核

from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError

from config.settings import ensure_dirs, settings
from ingest.chunker.router import chunk_document
from ingest.classify_pdf import classify_pdf
from ingest.extractors.law_detector import detect_laws, primary_law_to_ref
from ingest.extractors.metadata import derive_metadata, infer_kind
from ingest.normalizer import normalize
from ingest.parsers.doc_converter import ConversionError, convert_doc_to_docx
from ingest.parsers.docx_parser import parse_docx
from ingest.parsers.pdf_ocr import OcrError, parse_pdf_ocr
from ingest.parsers.pdf_table import parse_pdf_with_tables
from ingest.parsers.pdf_text import parse_pdf_text
from ingest.parsers.txt_parser import parse_txt
from ingest.parsers.violation_pdf import parse_violation_pdf


# 統計用
class Stats:
    def __init__(self):
        self.by_ext = {"txt": 0, "docx": 0, "doc": 0, "pdf": 0}
        self.pdf_types = {"normal_text": 0, "has_tables": 0, "needs_ocr": 0,
                          "empty": 0, "encrypted": 0, "corrupted": 0}
        self.ocr_success = 0
        self.ocr_failed = 0
        self.total_chunks = 0
        self.total_violations = 0
        self.failed_files = 0
        self.law_refs = 0


def iter_input_files(root: Path) -> Iterable[Path]:
    """遞迴掃所有可處理的檔案。"""
    exts = {".txt", ".docx", ".doc", ".pdf"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
            yield p


def is_violation_pdf(path: Path) -> bool:
    """判斷是否為北市違規月報。"""
    return ("違規廣告" in path.name) and path.suffix.lower() == ".pdf" \
        and "處罰案件" in path.name


def parse_one_file(path: Path, stats: Stats) -> tuple[dict | None, dict | None]:
    """解析單一檔案,回傳 (parsed_result, failed_record)。

    若成功,parsed_result = {"text": ..., "tables": [...], "meta": {...}}
    若失敗,failed_record = {...}
    其中一個會是 None。
    """
    ext = path.suffix.lower().lstrip(".")
    stats.by_ext[ext] = stats.by_ext.get(ext, 0) + 1

    try:
        if ext == "txt":
            result = parse_txt(path)
            return result, None

        if ext == "docx":
            try:
                result = parse_docx(path)
                return result, None
            except DocxPackageNotFoundError:
                # Binary .doc with .docx extension — fall through to DOC conversion
                docx_path = convert_doc_to_docx(path)
                result = parse_docx(docx_path)
                result["meta"]["converted_from"] = "binary_doc_renamed"
                return result, None

        if ext == "doc":
            # 先轉 docx
            docx_path = convert_doc_to_docx(path)
            result = parse_docx(docx_path)
            result["meta"]["converted_from"] = "doc"
            return result, None

        if ext == "pdf":
            # PDF 三狀況分流
            cls = classify_pdf(path)
            stats.pdf_types[cls["type"]] = stats.pdf_types.get(cls["type"], 0) + 1

            if cls["type"] == "normal_text":
                result = parse_pdf_text(path)
                result["meta"].update({"pdf_type": "normal_text"})
                return result, None

            if cls["type"] == "has_tables":
                result = parse_pdf_with_tables(path)
                result["meta"].update({"pdf_type": "has_tables"})
                return result, None

            if cls["type"] == "needs_ocr":
                try:
                    result = parse_pdf_ocr(path)
                    # OCR 後再次檢查是否有東西
                    if not result["text"].strip() or len(result["text"]) < 50:
                        stats.ocr_failed += 1
                        return None, {
                            "source_path": str(path),
                            "file_name": path.name,
                            "file_ext": ext,
                            "page_count": cls["n_pages"],
                            "failure_reason": "ocr_no_text",
                            "note": "OCR 後仍無文字(< 50 字),原始檔可能為純圖或文字無法辨識",
                        }
                    stats.ocr_success += 1
                    result["meta"].update({"pdf_type": "ocr", "is_ocr": True})
                    return result, None
                except OcrError as e:
                    stats.ocr_failed += 1
                    return None, {
                        "source_path": str(path),
                        "file_name": path.name,
                        "file_ext": ext,
                        "page_count": cls["n_pages"],
                        "failure_reason": "ocr_error",
                        "note": str(e),
                    }

            # empty / encrypted / corrupted
            return None, {
                "source_path": str(path),
                "file_name": path.name,
                "file_ext": ext,
                "page_count": cls["n_pages"],
                "failure_reason": cls["type"],
                "note": cls.get("error") or f"PDF classified as {cls['type']}",
            }

    except ConversionError as e:
        return None, {
            "source_path": str(path),
            "file_name": path.name,
            "file_ext": ext,
            "page_count": None,
            "failure_reason": "conversion_failed",
            "note": str(e),
        }
    except Exception as e:
        return None, {
            "source_path": str(path),
            "file_name": path.name,
            "file_ext": ext,
            "page_count": None,
            "failure_reason": "parser_exception",
            "note": f"{type(e).__name__}: {e}",
        }


def process_chunks(parsed: dict, path: Path, stats: Stats) -> list[dict]:
    """把 parsed 的 text 切碎、加 metadata、偵測法條,回傳 chunk dicts。"""
    md = derive_metadata(path, root=str(settings.raw_dir))
    md["kind"] = infer_kind(path.name, md.get("primary_law"))
    md["is_ocr"] = parsed["meta"].get("is_ocr", False)
    md["has_table"] = len(parsed.get("tables", [])) > 0

    raw_text = parsed["text"]
    # 對非 hash/qa 格式的內容跑 normalize(hash/qa 自己會處理)
    chunks_text, strategy = chunk_document(raw_text, path.name)

    if not chunks_text and raw_text.strip():
        # 切不出來但有文字 → 整段當一個 chunk(避免完全丟失)
        chunks_text = [normalize(raw_text)[:settings.chunk_max_len]]
        strategy = "whole"

    out: list[dict] = []
    for ct in chunks_text:
        # 法條偵測(網狀)
        law_refs = detect_laws(ct, md.get("primary_law"))
        # 把 primary_law(從路徑)當作 fallback role=primary 加入
        primary_ref = primary_law_to_ref(md.get("primary_law") or "")
        if primary_ref and not any(
            r.law_name == primary_ref.law_name and r.article_full == primary_ref.article_full
            for r in law_refs
        ):
            law_refs.insert(0, primary_ref)

        stats.law_refs += len(law_refs)

        out.append({
            "text": ct,
            "char_len": len(ct),
            "chunk_strategy": strategy,
            # metadata
            "category": md.get("category"),
            "primary_law": md.get("primary_law"),
            "subtopic": md.get("subtopic"),
            "document": md.get("document"),
            "kind": md.get("kind"),
            "is_ocr": md.get("is_ocr", False),
            "has_table": md.get("has_table", False),
            "source_path": md.get("source_path"),
            # 法條(多對多)
            "law_refs": [asdict(r) for r in law_refs],
        })
    return out


def _worker(path_str: str) -> dict:
    """子程序：解析單一檔案，回傳序列化結果（不寫檔）。"""
    path = Path(path_str)
    stats = Stats()  # 每個 worker 有自己的 stats，最後在主程序合併

    if is_violation_pdf(path):
        cases = parse_violation_pdf(path)
        return {"kind": "violations", "data": cases, "path": path_str}

    parsed, failed = parse_one_file(path, stats)
    if failed is not None:
        md = derive_metadata(path, root=str(settings.raw_dir))
        failed["inferred_law"] = md.get("primary_law")
        failed["inferred_topic"] = md.get("subtopic")
        return {"kind": "failed", "data": failed, "path": path_str}

    chunks = process_chunks(parsed, path, stats)
    return {
        "kind": "ok",
        "chunks": chunks,
        "path": path_str,
        "stats": {
            "by_ext": dict(stats.by_ext),
            "pdf_types": dict(stats.pdf_types),
            "ocr_success": stats.ocr_success,
            "ocr_failed": stats.ocr_failed,
            "law_refs": stats.law_refs,
        },
    }


def main():
    ensure_dirs()

    chunks_out = settings.processed_dir / "chunks.jsonl"
    violations_out = settings.processed_dir / "violations.jsonl"
    failed_out = settings.processed_dir / "failed_files.jsonl"

    stats = Stats()
    chunks_fp = chunks_out.open("w", encoding="utf-8")
    violations_fp = violations_out.open("w", encoding="utf-8")
    failed_fp = failed_out.open("w", encoding="utf-8")

    files = list(iter_input_files(settings.raw_dir))
    print(f"\n發現 {len(files)} 個可處理檔案（{WORKERS} 個 worker 平行處理）\n")

    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(_worker, str(p)): p for p in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="解析中"):
                try:
                    result = future.result()
                except Exception as e:
                    path = futures[future]
                    failed_rec = {
                        "source_path": str(path),
                        "file_name": path.name,
                        "file_ext": path.suffix.lstrip("."),
                        "page_count": None,
                        "failure_reason": "worker_exception",
                        "note": f"{type(e).__name__}: {e}",
                    }
                    failed_fp.write(json.dumps(failed_rec, ensure_ascii=False) + "\n")
                    stats.failed_files += 1
                    continue

                if result["kind"] == "violations":
                    for c in result["data"]:
                        violations_fp.write(json.dumps(c, ensure_ascii=False) + "\n")
                        stats.total_violations += 1

                elif result["kind"] == "failed":
                    failed_fp.write(json.dumps(result["data"], ensure_ascii=False) + "\n")
                    stats.failed_files += 1

                else:  # ok
                    for chunk in result["chunks"]:
                        chunks_fp.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        stats.total_chunks += 1
                    # 合併 worker stats
                    ws = result["stats"]
                    for k, v in ws["by_ext"].items():
                        stats.by_ext[k] = stats.by_ext.get(k, 0) + v
                    for k, v in ws["pdf_types"].items():
                        stats.pdf_types[k] = stats.pdf_types.get(k, 0) + v
                    stats.ocr_success += ws["ocr_success"]
                    stats.ocr_failed  += ws["ocr_failed"]
                    stats.law_refs    += ws["law_refs"]

    finally:
        chunks_fp.close()
        violations_fp.close()
        failed_fp.close()

    # === 報告 ===
    print("\n" + "=" * 60)
    print("INGEST Step 1 完成")
    print("=" * 60)
    print("\n📂 處理檔案(按副檔名):")
    for ext, n in stats.by_ext.items():
        print(f"   .{ext:5s}: {n}")

    print("\n📄 PDF 分類:")
    for t, n in stats.pdf_types.items():
        print(f"   {t:15s}: {n}")
    print(f"   OCR 成功 / 失敗: {stats.ocr_success} / {stats.ocr_failed}")

    print("\n📊 結果:")
    print(f"   總 chunks         : {stats.total_chunks}")
    print(f"   總違規案例         : {stats.total_violations}")
    print(f"   失敗檔案數         : {stats.failed_files}")
    print(f"   偵測到的法條引用    : {stats.law_refs}")
    if stats.total_chunks > 0:
        print(f"   平均法條/chunk     : {stats.law_refs / stats.total_chunks:.2f}")

    print("\n📁 輸出檔:")
    print(f"   {chunks_out}")
    print(f"   {violations_out}")
    print(f"   {failed_out}")

    if stats.failed_files > 0:
        print(f"\n⚠️  有 {stats.failed_files} 個檔案無法處理。")
        print("   執行 `make failed` 或 `python scripts/check_failed_files.py` 檢視清單。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中斷")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
