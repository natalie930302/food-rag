"""
從考選部「營養師」國考「食品衛生與安全」歷屆試題(單選題 + 官方標準答案)建立評估集。

為什麼用這個:專案原本的評估集是 24 題手寫 + 84 題 LLM 生成,都是「看著語料出題」,
無法回答「真實專家怎麼問」。國考題是命題委員寫的、有官方標準答案、依《著作權法》第 9 條
不受著作權保護,而且是選擇題——可以直接量「系統答對沒有」,不只是「有沒有撈到 chunk」。

來源:考選部考畢試題查詢平臺 https://wwwq.moex.gov.tw/exam/
  試題 wHandExamQandA_File.ashx?t=Q&code=<年+030>&c=<類科>&s=<科目>&q=1,答案 t=S
  109–114 年第一次專技高考營養師,科目「食品衛生與安全」(109–112:申論 + 40 題單選;113–114:50 題單選)

兩種 PDF 格式:
  舊(109–112):題號後接空白;選項編號是私用區字元 \\ue18c–\\ue18f(①②③④),文字抽取後常在同一行
  新(113–114):「1.題幹」「A.選項」逐行
答案卷:「題號 第1題 第2題…/答案 B C…」或「題號 01 02…/答案 Ｄ Ｄ…」(全形)

不是每題都跟語料有關(微生物、毒理、加工學的題目答案不在食藥署法規/指引裡)。刻意不做人工
篩選(那是主觀判斷):每題附 relevance_hint(可重現的關鍵字啟發式),eval_exam.py 全部 260 題
都跑、依 hint 分組報告,拒答算錯——RAG 只有真的找到依據才會贏過閉卷 LLM。

用法:python scripts/build_exam_set.py <含 diet_<code>_Q.pdf / _S.pdf 的資料夾>
  → eval/exam_questions.json(全部 260 題 + relevance_hint;eval/eval_exam.py 直接用,不做人工篩選)
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).parent.parent
MARKERS = {"": "A", "": "B", "": "C", "": "D"}
FULLWIDTH = {"Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D"}

# 跟語料(食藥署法規、指引、問答集、標準)有關的訊號;沒有這些字眼的多半是微生物/毒理/加工題
REG_CUES = ("法", "條", "規定", "規範", "標示", "標準", "登錄", "罰", "公告", "準則", "辦法", "宣稱", "廣告",
            "添加物", "限量", "應遵行", "主管機關", "許可", "登記", "查驗", "追溯", "衛生福利部", "食藥署",
            "食品安全管制系統", "HACCP", "GHP", "營養", "包裝", "容器", "檢驗", "業者", "輸入", "產品責任")


def extract_pages(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)
    text = "\n".join(texts)
    for glyph, letter in MARKERS.items():
        text = text.replace(glyph, f"\n{letter}.")
    return text


def parse_answers(text: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith("題號"):
            continue
        nums = [int(n) for n in re.findall(r"第?(\d+)題?", line)]
        if i + 1 < len(lines) and lines[i + 1].startswith("答案"):
            letters = [FULLWIDTH.get(ch, ch) for ch in lines[i + 1][2:].replace(" ", "") if ch.strip()]
            for n, a in zip(nums, letters):
                if a in "ABCD":
                    answers[n] = a
    return answers


def parse_questions(text: str) -> list[dict]:
    # 只取單選題部分(舊格式從「乙、測驗題部分」開始)
    if "測驗題部分" in text:
        text = text.split("測驗題部分", 1)[1]
    # 去掉頁首頁尾雜訊
    text = re.sub(r"代號：\d+\n頁次：\d+－\d+\n?", "", text)
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]

    questions: list[dict] = []
    cur: dict | None = None
    opt: str | None = None
    q_re = re.compile(r"^(\d{1,2})[\.\s]\s*(.*)$")
    o_re = re.compile(r"^([ABCD])\.\s*(.*)$")

    for line in lines:
        m_o = o_re.match(line)
        m_q = q_re.match(line)
        if m_q and (cur is None or int(m_q.group(1)) == cur["no"] + 1) and not m_o:
            if cur:
                questions.append(cur)
            cur = {"no": int(m_q.group(1)), "stem": m_q.group(2).strip(), "options": {}}
            opt = None
        elif m_o and cur is not None:
            opt = m_o.group(1)
            cur["options"][opt] = m_o.group(2).strip()
        elif cur is not None:
            if opt:
                cur["options"][opt] += line.strip()
            else:
                cur["stem"] += line.strip()
    if cur:
        questions.append(cur)
    return questions


def relevance_hint(stem: str, options: dict) -> bool:
    blob = stem + "".join(options.values())
    return any(c in blob for c in REG_CUES)


def main(folder: str):
    folder = Path(folder)
    out: list[dict] = []
    for q_pdf in sorted(folder.glob("diet_*_Q.pdf")):
        code = q_pdf.stem.split("_")[1]
        s_pdf = folder / f"diet_{code}_S.pdf"
        if not s_pdf.exists():
            print(f"跳過 {q_pdf.name}:沒有答案卷")
            continue
        year = int(code[:3])
        qs = parse_questions(extract_pages(q_pdf))
        answers = parse_answers(extract_pages(s_pdf))
        ok = 0
        for q in qs:
            if len(q["options"]) != 4 or q["no"] not in answers:
                continue
            ok += 1
            out.append({
                "id": f"exam-{year}-{q['no']:02d}",
                "year": year, "no": q["no"],
                "stem": q["stem"], "options": q["options"], "answer": answers[q["no"]],
                "relevance_hint": relevance_hint(q["stem"], q["options"]),
                "source": f"考選部 {year} 年第一次專技高考營養師「食品衛生與安全」第 {q['no']} 題",
            })
        print(f"{year}: 解析到 {len(qs)} 題,完整(4 選項 + 答案){ok} 題,答案卷 {len(answers)} 題")

    dst = ROOT / "eval" / "exam_questions.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    hinted = sum(q["relevance_hint"] for q in out)
    print(f"\n共 {len(out)} 題 → {dst};關鍵字啟發式標為「可能跟語料相關」的 {hinted} 題(eval_exam.py 會分組報告,不做人工篩選)")


if __name__ == "__main__":
    main(sys.argv[1])
