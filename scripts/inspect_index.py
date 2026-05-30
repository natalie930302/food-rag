"""顯示索引統計與健康狀況。"""
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

    print("=" * 60)
    print("索引統計")
    print("=" * 60)

    # 基本計數
    for table in ("chunks", "chunk_laws", "violations", "failed_files"):
        n = cur.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        print(f"  {table:20s}: {n}")

    # by kind
    print("\n--- chunks by kind ---")
    for r in cur.execute("SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind ORDER BY n DESC"):
        print(f"  {r['kind'] or '(none)':15s}: {r['n']}")

    # by primary_law top 15
    print("\n--- chunks by primary_law (top 15) ---")
    for r in cur.execute(
        """SELECT primary_law, COUNT(*) AS n FROM chunks
           WHERE primary_law IS NOT NULL GROUP BY primary_law
           ORDER BY n DESC LIMIT 15"""
    ):
        print(f"  {r['primary_law']:20s}: {r['n']}")

    # 法條偵測 top
    print("\n--- 法條偵測 top 10 ---")
    for r in cur.execute(
        """SELECT law_name, article_full, COUNT(DISTINCT chunk_id) AS n
           FROM chunk_laws GROUP BY law_name, article_full
           ORDER BY n DESC LIMIT 10"""
    ):
        print(f"  {r['law_name']}第{r['article_full']}條: {r['n']} chunks")

    # OCR / 表格
    ocr = cur.execute("SELECT COUNT(*) AS n FROM chunks WHERE is_ocr = 1").fetchone()["n"]
    tab = cur.execute("SELECT COUNT(*) AS n FROM chunks WHERE has_table = 1").fetchone()["n"]
    print(f"\n--- 特殊屬性 ---")
    print(f"  OCR 來源 chunks   : {ocr}")
    print(f"  含表格 chunks     : {tab}")

    # 違規案例統計
    n_viol = cur.execute("SELECT COUNT(*) AS n FROM violations").fetchone()["n"]
    if n_viol:
        s = cur.execute("SELECT SUM(penalty_twd) AS s FROM violations").fetchone()["s"] or 0
        avg = s / n_viol
        print(f"\n--- 違規案例 ---")
        print(f"  總案數    : {n_viol}")
        print(f"  總罰鍰    : NT${s:,}")
        print(f"  平均罰鍰  : NT${avg:,.0f}")
        print("  法條分布(top 5):")
        for r in cur.execute(
            """SELECT article_no, COUNT(*) AS n FROM violations
               WHERE article_no IS NOT NULL GROUP BY article_no
               ORDER BY n DESC LIMIT 5"""
        ):
            print(f"    食安法第{r['article_no']}條: {r['n']} 案")

    # 失敗檔
    n_fail = cur.execute("SELECT COUNT(*) AS n FROM failed_files").fetchone()["n"]
    if n_fail:
        print(f"\n--- 失敗檔案 ---")
        for r in cur.execute(
            """SELECT failure_reason, COUNT(*) AS n FROM failed_files
               GROUP BY failure_reason ORDER BY n DESC"""
        ):
            print(f"  {r['failure_reason']:25s}: {r['n']}")

    conn.close()


if __name__ == "__main__":
    main()
