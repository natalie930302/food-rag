"""列出所有無法處理的檔案,幫助人工後續處理。"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def main():
    if not settings.db_path.exists():
        print(f"找不到 {settings.db_path},請先跑 `make ingest`")
        sys.exit(1)

    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) AS n FROM failed_files").fetchone()["n"]
    if total == 0:
        print("✅ 沒有失敗的檔案。所有檔案皆已成功處理。")
        return

    print(f"\n共 {total} 個檔案無法處理\n")
    print("=" * 80)

    # 按 failure_reason 分組
    reasons = cur.execute(
        """SELECT failure_reason, COUNT(*) AS n FROM failed_files
           GROUP BY failure_reason ORDER BY n DESC"""
    ).fetchall()
    print("\n失敗原因分布:")
    for r in reasons:
        print(f"  {r['failure_reason']:25s}: {r['n']}")

    print("\n" + "=" * 80)
    print("詳細清單:")
    print("=" * 80)

    for r in cur.execute(
        """SELECT file_name, failure_reason, page_count,
                  inferred_law, inferred_topic, note
           FROM failed_files ORDER BY failure_reason, file_name"""
    ):
        print(f"\n📄 {r['file_name']}")
        print(f"   原因: {r['failure_reason']}")
        if r['page_count']:
            print(f"   頁數: {r['page_count']}")
        if r['inferred_law']:
            print(f"   推測法條: {r['inferred_law']}")
        if r['inferred_topic']:
            print(f"   推測主題: {r['inferred_topic']}")
        if r['note']:
            note = r['note'][:120] + "..." if len(r['note']) > 120 else r['note']
            print(f"   備註: {note}")

    conn.close()


if __name__ == "__main__":
    main()
