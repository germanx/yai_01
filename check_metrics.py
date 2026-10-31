import re
import json

EXPECTED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags",
}

MIN_DESCRIPTION_LENGTH = 120 
MAX_DESCRIPTION_LENGTH = 700 


class ResponseValidationError(ValueError):
    """Ошибка парсинга или проверки ответа модели."""


def count_sentences(text):
    """Число предложений в тексте. Точка внутри числа концом предложения не считается."""
    return len([s for s in re.split(r"[.!?]+(?!\d)", text) if s.strip()]) 


def parse_and_validate(content: str, product_id_req: str) -> dict[str, str]:
    """Преобразовать JSON-ответ и проверить его контракт."""

    ret = {"product_id": product_id_req}

    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        return ret | {'error': f"Ответ не является корректным JSON: {error.msg}"}

    if not isinstance(data, dict):
        return ret | {'error': "На верхнем уровне должен находиться JSON-объект"}

    received_fields = set(data)
    missing_fields = EXPECTED_FIELDS - received_fields
    unexpected_fields = received_fields - EXPECTED_FIELDS

    if missing_fields:
        return ret | {'error': "Отсутствуют обязательные поля: "
            f"{sorted(missing_fields)}"
        }

    if unexpected_fields:
        return ret | {'error': "Отсутствуют обязательные поля: "
            "Получены лишние поля: "
            f"{sorted(unexpected_fields)}"
        }

    product_id = data['product_id']
    if product_id.strip() != product_id_req.strip():
        return ret | {'error': "Поле product_id не соответствует входным параметрам"}

    description = data["description"]

    if not isinstance(description, str):
        return ret | {'error': "Поле description должно быть строкой"}

    description = description.strip()

    if not description:
        return ret | {'error': "Поле description не должно быть пустым"}
      
    i = count_sentences(description)
    if i < 2 or i > 5:
        return ret | {'error': "Description must contain 2-5 sentences"}

    if len(description) < MIN_DESCRIPTION_LENGTH:
        return ret | {'error': "Отсутствуют обязательные поля: "
            "Поле description содержит больше "
            f"{MAX_DESCRIPTION_LENGTH} символов"
        }

    if len(description) > MAX_DESCRIPTION_LENGTH:
        return ret | {'error': "Поле description содержит меньше "
            f"{MAX_DESCRIPTION_LENGTH} символов"
        }

    pros = data['pros']
    if len(pros) < 2 or len(pros) > 5:
       return ret | {'error': "Поле pros must contain 2-5 values"}

    cons = data['cons']
    if len(cons) < 1 or len(cons) > 3:
       return ret | {'error': "Поле cons must contain 1-3 values"}
    
    tags = data['tags']
    if len(tags) < 3 or len(tags) > 8:
       return ret | {'error': "Поле tags must contain 3-8 values"}
    
    return ret


from typing import List
from typing_extensions import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class CardRequest(BaseModel):
    """Product Card"""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=50)
    description: Annotated[str, Field(min_length=120, max_length=700), AfterValidator(count_sentences)]
    pros: Annotated[List[str], Field(min_length=2, max_length=5)]
    cons: Annotated[List[str], Field(min_length=1, max_length=3)]
    tags: Annotated[List[str], Field(min_length=3, max_length=8)]


    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
      """Удалить внешние пробелы и проверить описание."""
      value = value.strip()

      if not value:
        raise ValueError(
            "Поле summary не должно быть пустым"
        )
      
      i = count_sentences(value)
      if i < 2 or i > 5:
        raise ValueError("Description must contain 2-5 sentences")
      
      return value


    @field_validator('pros')
    @classmethod
    def check_pros(cls, v: List[str]) -> List[str]:
      for i in v:
        if not i.strip():
          raise ValueError("Элементы pros не могут быть пустыми строками")
      return v


    @field_validator('cons')
    @classmethod
    def check_cons(cls, v: List[str]) -> List[str]:
      for i in v:
        if not i.strip():
          raise ValueError("Элементы cons не могут быть пустыми строками")
      return v


    @field_validator('tags')
    @classmethod
    def check_tags(cls, v: List[str]) -> List[str]:
      for i in v:
        if not i.strip():
          raise ValueError("Элементы tags не могут быть пустыми строками")
      return v
    