from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ---- Model Configuration ----
    MODEL_NAME: str = "deepseek-chat"
    DEEPSEEK_API_KEY: str

    # --- MongoDB Configuration ---
    MONGO_URI: str = Field(
        default="mongodb+srv://llmragchatbot:llmragchatbot@llmragchatbot.zipta.mongodb.net/",
        description="Connection URI for the local MongoDB Atlas instance.",
    )
    MONGO_DB_NAME: str = "philoagents"
    MONGO_STATE_CHECKPOINT_COLLECTION: str = "philosopher_state_checkpoints"
    MONGO_STATE_WRITES_COLLECTION: str = "philosopher_state_writes"
    MONGO_LONG_TERM_MEMORY_COLLECTION: str = "philosopher_long_term_memory"    

    # ---- Agent Configuration ----
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 30
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    # --- RAG Configuration ---
    RAG_TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_TEXT_EMBEDDING_MODEL_DIM: int = 384
    RAG_TOP_K: int = 3
    RAG_DEVICE: str = "cpu"
    RAG_CHUNK_SIZE: int = 256

settings = Settings()
print(settings.model_dump())