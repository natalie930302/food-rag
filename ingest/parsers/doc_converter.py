"""把 .doc 用 LibreOffice headless 轉成 .docx。

Windows 備援策略:
  1. 先嘗試設定路徑(settings.libreoffice_bin)
  2. 若找不到,自動掃 Windows 常見安裝路徑
  3. 若 LibreOffice 完全不存在,改用 Microsoft Word COM 自動化(需裝 pywin32)
"""
import subprocess
import sys
from pathlib import Path

from config.settings import settings


class ConversionError(Exception):
    pass


_WINDOWS_LO_PATHS = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def _find_libreoffice() -> str:
    """回傳可用的 LibreOffice 執行檔路徑。"""
    configured = settings.libreoffice_bin
    # 使用者明確指定路徑時直接用
    if configured not in ("libreoffice", "soffice"):
        return configured
    if sys.platform == "win32":
        for p in _WINDOWS_LO_PATHS:
            if Path(p).exists():
                return p
    return configured


def _convert_with_win32com(doc_path: Path, out_dir: Path) -> Path:
    """Windows 備援:用 Microsoft Word COM 把 .doc 存成 .docx。"""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        raise ConversionError(
            "LibreOffice 與 pywin32 都找不到。\n"
            "請擇一安裝:\n"
            "  LibreOffice: https://www.libreoffice.org/download/\n"
            "  pywin32 (需已裝 Microsoft Word): pip install pywin32"
        )

    out_path = out_dir / (doc_path.stem + ".docx")
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(doc_path.resolve()))
        doc.SaveAs2(str(out_path.resolve()), FileFormat=16)  # 16 = wdFormatXMLDocument (.docx)
        doc.Close()
        word.Quit()
    except Exception as e:
        raise ConversionError(f"Word COM 轉換失敗: {e}")

    if not out_path.exists():
        raise ConversionError(f"COM 轉換完成但找不到輸出檔: {out_path}")
    return out_path


def convert_doc_to_docx(doc_path: Path | str, out_dir: Path | None = None) -> Path:
    """轉換 .doc → .docx,回傳輸出檔路徑。

    若 out_dir 為 None,使用 settings.converted_dir。
    若已存在轉好的檔,直接回傳(避免重複轉)。
    """
    doc_path = Path(doc_path)
    if out_dir is None:
        out_dir = settings.converted_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (doc_path.stem + ".docx")
    if out_path.exists():
        return out_path

    libreoffice_bin = _find_libreoffice()
    cmd = [
        libreoffice_bin,
        "--headless",
        "--convert-to", "docx",
        "--outdir", str(out_dir),
        str(doc_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise ConversionError(
                f"libreoffice exit {result.returncode}: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise ConversionError(f"timeout converting {doc_path}")
    except FileNotFoundError:
        if sys.platform == "win32":
            return _convert_with_win32com(doc_path, out_dir)
        raise ConversionError(
            f"libreoffice not found (path={libreoffice_bin}). "
            "Install: https://www.libreoffice.org/download/"
        )

    if not out_path.exists():
        raise ConversionError(f"conversion succeeded but output not found: {out_path}")
    return out_path
