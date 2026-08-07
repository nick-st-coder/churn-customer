from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    threshold: float = 0.36

    class Config:
        env_file = ".env"
        
settings = Settings()        