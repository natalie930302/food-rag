.PHONY: install unzip ingest parse build-index run run-prod test stats failed clean help

help:
	@echo "Available commands:"
	@echo "  make install     - 建立 venv + 安裝套件"
	@echo "  make unzip       - 解壓 data/raw/zips/ 內的 zip 檔"
	@echo "  make ingest      - 跑完整 ingest pipeline (parse + build-index)"
	@echo "  make parse       - 只跑 step1 (解析所有檔案 → JSONL)"
	@echo "  make build-index - 只跑 step2 (編碼 + 寫 SQLite + FAISS)"
	@echo "  make run         - 啟動 API (dev mode)"
	@echo "  make run-prod    - 啟動 API (production)"
	@echo "  make stats       - 印出索引統計"
	@echo "  make failed      - 印出無法處理的檔案清單"
	@echo "  make test        - 跑 pytest"
	@echo "  make clean       - 清掉 data/index 與 data/processed"

install:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

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
