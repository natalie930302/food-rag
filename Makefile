.PHONY: install unzip ingest parse build-index run run-prod test lint eval eval-retrieval eval-agent eval-harness eval-report stats failed clean help

help:
	@echo "Available commands:"
	@echo "  make install     - 建立 venv + 安裝套件"
	@echo "  make build-ui    - 打包前端 (需先安裝 Node.js)"
	@echo "  make unzip       - 解壓 data/raw/zips/ 內的 zip 檔"
	@echo "  make ingest      - 跑完整 ingest pipeline (parse + build-index)"
	@echo "  make parse       - 只跑 step1 (解析所有檔案 → JSONL)"
	@echo "  make build-index - 只跑 step2 (編碼 + 寫 SQLite + FAISS)"
	@echo "  make run         - 啟動 API (dev mode, 含前端 UI)"
	@echo "  make run-prod    - 啟動 API (production, 含前端 UI)"
	@echo "  make stats       - 印出索引統計"
	@echo "  make failed      - 印出無法處理的檔案清單"
	@echo "  make test        - 跑 pytest"
	@echo "  make lint        - ruff 靜態檢查"
	@echo "  make eval        - 重跑全部檢索/agent 評估並產出 eval/RESULTS.md"
	@echo "  make eval-harness - 多步問題集 + harness 消融(較貴)"
	@echo "  make clean       - 清掉 data/index 與 data/processed"

install:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

build-ui:
	cd ../food-rag-ui && npm install && npm run build

unzip:
	python scripts/unzip_uploads.py

ingest: parse build-index

parse:
	python -u -m ingest.step1_parse

build-index:
	python -u -m ingest.step2_build_index

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000

stats:
	python scripts/inspect_index.py

failed:
	python scripts/check_failed_files.py

test:
	pytest tests/ -v

clean:
	rm -rf data/index/* data/processed/* data/converted/*
	@echo "Cleaned data/index, data/processed, data/converted"

lint:
	ruff check app ingest eval tests scripts

# 重跑全部檢索評估並產出 eval/RESULTS.md(需要 data/index/ 與 OPENAI_API_KEY;
# agent 相關評估會呼叫 gpt-4o-mini,100 題約數十元台幣)
eval: eval-retrieval eval-agent eval-report

eval-retrieval:
	python eval/eval_retrieval.py
	python eval/eval_reranking.py
	python eval/tune_confidence_threshold.py
	python eval/eval_corrective.py

eval-agent:
	python eval/eval_agentic.py
	python eval/eval_tool_agent_drift_check.py
	python eval/eval_citation_verifier.py

# harness 專屬評估:多步問題集 + 消融(較貴,約 1.5 小時、gpt-4o-mini 數十元台幣)
eval-harness:
	python eval/eval_multihop.py
	python eval/eval_harness_ablation.py

eval-report:
	python eval/summarize.py
