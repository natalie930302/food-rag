"""把 data/raw/zips/ 內所有 zip 解壓到 data/raw/。

支援的 zip 內檔名編碼:UTF-8(zip 內的 flag_bits 0x800)。
若是 cp437/big5 編碼,會嘗試修正。
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings, ensure_dirs


def fix_chinese_filename(name: str) -> str:
    """處理舊版 zip 用 cp437 偽編碼中文檔名的情況。"""
    try:
        # 試:重新用 cp437 編碼,再用 utf-8 解碼
        fixed = name.encode("cp437").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            fixed = name.encode("cp437").decode("big5")
            return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            return name


def extract_zip(zip_path: Path, dest: Path):
    print(f"\n解壓 {zip_path.name} → {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            # 檢查是否為 UTF-8 編碼
            if info.flag_bits & 0x800:
                # 已是 UTF-8,直接用
                pass
            else:
                # 嘗試修正
                info.filename = fix_chinese_filename(info.filename)
            zf.extract(info, str(dest))
    print(f"  完成")


def main():
    ensure_dirs()
    zips_dir = settings.raw_dir / "zips"
    if not zips_dir.exists():
        print(f"找不到 {zips_dir}")
        print("請把原始 zip 放到此目錄,例如:")
        print(f"  cp 食藥署.zip {zips_dir}/")
        sys.exit(1)

    zips = sorted(zips_dir.glob("*.zip"))
    if not zips:
        print(f"{zips_dir} 下沒有 zip 檔")
        sys.exit(1)

    # 若內層還有 zip,先全部解到 raw_dir,再遞迴解開
    for z in zips:
        extract_zip(z, settings.raw_dir)

    # 第二輪:處理巢狀 zip(例如最外層那個 drive-download 包了三個內部 zip)
    while True:
        inner_zips = [
            p for p in settings.raw_dir.rglob("*.zip")
            if p.parent != zips_dir
        ]
        if not inner_zips:
            break
        for z in inner_zips:
            extract_zip(z, z.parent)
            z.unlink()  # 解完刪掉避免下次再處理

    print("\n所有 zip 已解壓完成。資料結構:")
    for top in sorted(settings.raw_dir.iterdir()):
        if top.is_dir() and top.name != "zips":
            n = sum(1 for _ in top.rglob("*") if _.is_file())
            print(f"  📁 {top.name}/  ({n} 檔)")


if __name__ == "__main__":
    main()
