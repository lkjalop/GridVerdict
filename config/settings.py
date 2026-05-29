from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_SECRETS = {"change_me", "secret", "changeme", "password", "test", "dev"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://gv:changeme@localhost/gridverdict"
    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    nemweb_cache_dir: str = "./cache/nemweb"
    nemweb_base_url: str = "https://nemweb.com.au"
    nemweb_poll_seconds: int = 300  # 5-minute dispatch cycle

    db_echo: bool = False
    log_level: str = "INFO"
    gridverdict_dev_no_auth: bool = False  # True = skip auth for localhost dev

    # Freshness thresholds (seconds)
    live_dispatch_max_age_s: int = 180
    predispatch_max_age_s: int = 900
    notices_max_age_s: int = 120
    nem_news_max_age_s: int = 600
    weather_max_age_s: int = 900

    # Model profile
    model_profile: str = "cost_optimized"
    enable_experimental_sequence_forecasters: bool = True
    default_forecast_horizon_intervals: int = 48   # 48 × 5min = 4-hour ahead forecast

    # LLM decomposer — Ollama default, Claude fallback
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_timeout_s: float = 30.0
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"   # cheapest capable model for decomp
    decomposer_backend: str = "ollama"   # "ollama" | "claude" | "rule_based"

    # Background scheduler
    aemo_dispatch_poll_s: int = 300    # 5-min dispatch cycle
    aemo_notices_poll_s: int = 60      # 1-min notice check
    nem_news_poll_s: int = 300
    weather_poll_s: int = 300

    # Historical archive / market scope
    backfill_start_date: str = "2022-08-01"   # 2 years back from last accessible month (2024-07)
    archive_backfill_enabled: bool = False
    archive_bulk_backfill_enabled: bool = False
    archive_bulk_backfill_max_files: int | None = None   # None = no limit
    backfill_tables: str = (
        "DISPATCHPRICE,DISPATCHINTERCONNECTORRES,DISPATCHCONSTRAINT,DUDETAILSUMMARY,"
        "BIDDAYOFFER,BIDPEROFFER"
        # DISPATCH_UNIT_SOLUTION is participant-only on NEMWeb MMSDM; not publicly available
    )
    backfill_request_delay_s: float = 1.0   # polite delay between NEMWeb requests
    nem_only: bool = True

    # Public RSS market commentary. Comma-separated in env if overridden.
    nem_news_rss_urls: str = (
        "https://wattclarity.com.au/feed/,"
        "https://reneweconomy.com.au/feed/"
    )
    nem_news_keywords: str = (
        "nem,aemo,dispatch,wholesale electricity,renewables,battery,coal,gas,"
        "transmission,interconnector,price spike,lack of reserve,"
        "spot price,market price,energy market,constraint,outage,solar farm,wind farm,"
        "rooftop solar,frequency,contingency,forced outage,trip,fault"
    )


    # Redis — optional; empty string disables Redis and falls back to in-process
    redis_url: str = ""  # e.g. "redis://localhost:6379/0"
    redis_leader_lock_ttl_s: int = 30       # lock TTL before failover
    redis_leader_lock_heartbeat_s: int = 10  # heartbeat interval

    @model_validator(mode="after")
    def _check_jwt_secret(self) -> "Settings":
        """Reject unsafe JWT secrets in non-dev mode at startup time."""
        if not self.gridverdict_dev_no_auth:
            secret = self.jwt_secret
            if secret.lower() in _INSECURE_SECRETS or len(secret) < 32:
                raise ValueError(
                    "JWT_SECRET is insecure or missing. "
                    "Set a strong random JWT_SECRET (>= 32 chars) in your .env file. "
                    "For local development set GRIDVERDICT_DEV_NO_AUTH=true."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
