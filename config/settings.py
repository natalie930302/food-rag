"""集中管理所有設定,從 .env 讀取。"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 1500

    # Embedding
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    embed_batch_size: int = 32

    # OCR
    tesseract_lang: str = "chi_tra+eng"
    ocr_dpi: int = 200
    ocr_min_chars_per_page: int = 30

    # 路徑
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    converted_dir: Path = PROJECT_ROOT / "data" / "converted"

    # 外部工具
    libreoffice_bin: str = "libreoffice"
    poppler_path: str | None = None

    # API
    log_level: str = "INFO"

    # Chunking
    chunk_max_len: int = 600
    chunk_min_len: int = 30
    chunk_overlap: int = 50

    # 衍生路徑(常用)
    @property
    def db_path(self) -> Path:
        return self.index_dir / "chunks.db"

    @property
    def faiss_chunks_path(self) -> Path:
        return self.index_dir / "faiss_chunks.index"

    @property
    def faiss_cases_path(self) -> Path:
        return self.index_dir / "faiss_cases.index"



settings = Settings()


def ensure_dirs():
    """啟動時確保所有資料夾存在。"""
    for d in [settings.data_dir, settings.raw_dir, settings.processed_dir,
              settings.index_dir, settings.converted_dir]:
        d.mkdir(parents=True, exist_ok=True)
