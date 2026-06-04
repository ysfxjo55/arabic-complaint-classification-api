import re

from pydantic import BaseModel, field_validator


class ArabicInput(BaseModel):
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def normalize_arabic(cls, value: str) -> str:
        if not isinstance(value, str):
            return value
        
        value = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', value)
        
        value = re.sub(r'[أإآٱ]', 'ا', value)
        
        value = value.replace('ة', 'ه').replace('ى', 'ي')
        
        value = re.sub(r'\s+', ' ', value).strip()
        
        return value