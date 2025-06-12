from pydantic_config import SettingsModel


class Config(SettingsModel):
    price_per_token: int = 100
