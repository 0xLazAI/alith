from pydantic_config import SettingsModel


class Config(SettingsModel):
    file_id_cache_size: int = 100
    price_per_token: int = 1
