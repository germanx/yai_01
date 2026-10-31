from ollama import Client
from datetime import datetime
from pathlib import Path
import json

from compare_api import CardRequest, ResponseValidationError, parse_and_validate

MAX_ATTEMPTS = 3 
MAX_TOKENS = 1500
MAX_REVIEWS = 30

model='llama3.1:8b'
client = Client(host='http://localhost:11434')


def request_one(
    msg: list[dict], 
    params={"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350}
) -> tuple[str, int]:
  messages=[
    {"role": "system", "content": 
      """
        Generate a product card in Russian (utf-8) by example.
        Example:
        {
          "product_id": "P-1001",
          "description": "Компактные внутриканальные наушники с Bluetooth 5.3 и защитой IPX4. Держат заряд до 24 часов и надёжно сидят при активном движении. Подойдут для спорта и повседневного использования.",
          "pros": ["Чистый звук", "Удобная посадка", "Долгое время работы"],
          "cons": ["Быстро разряжаются на морозе"],
          "tags": ["наушники", "беспроводные", "спорт", "IPX4", "Bluetooth 5.3"]
        } 
        "description": 2-5 sentences,
        return only JSON
      """
    } 
  ] + msg
  # print('>>> messages', messages)

  response = client.chat(
      model=model,  
      messages=messages,
      options={
        'seed':               42,  
        'temperature':        params['temperature'],  
        'top_p':              params['top_p'],
        'max_tokens':         params['max_tokens'],   
        "top_k":              params['top_k'], 
        "repetition_penalty": params['repetition_penalty'],      
      },
      format=CardRequest.model_json_schema(),
  )

  content = response['message']['content']
  if not content:
    raise ResponseValidationError(
        "Empty response of model"
    )

  output_tokens = response.eval_count
  
  return content, output_tokens


#####################################
import time

def worming_up():
  item = {"product_id": "P-0026", "title": "Смартфон Grafo G9", "category": "Электроника / Смартфоны", "attributes": {"бренд": "Grafo", "экран": "6.7\"", "память": "256 ГБ", "оперативная память": "8 ГБ", "аккумулятор": "6000 мА·ч", "основная камера": "50 Мп"}, "reviews": ["Пользуюсь пару недель. Камера снимает достойно даже вечером", "Брал для себя. Звонки чистые, собеседника слышно хорошо", "В целом Быстро заряжается, за час почти полный", "Заказывал на подарок. Работает шустро, приложения не тормозят", "Брал для себя. Камера мылит фото при недостатке света", "Пользуюсь пару недель. К вечеру заряд просаживается быстрее ожидаемого", "Пользуюсь пару недель. Экран яркий, на солнце всё видно", "Пользуюсь пару недель. Экран яркий, на солнце всё видно", "Пользуюсь каждый день. Экран яркий, на солнце всё видно", "Быстро заряжается, за час почти полный", "В руке лежит удобно, не скользит", "Динамик тихий, в шумном месте плохо слышно", "Работает шустро, приложения не тормозят", "В руке лежит удобно, не скользит", "Коротко: Работает шустро, приложения не тормозят", "Купил на замену старому. Обычный телефон на каждый день, без сюрпризов", "Брал для себя. Батарея держит полтора дня без подзарядки", "В целом Работает шустро, приложения не тормозят"]}

  row_dict = [{"role": "user", "content": f"""{item}"""}]

  started = time.perf_counter()
  resp, tokens = request_one(row_dict)
  generation_time = time.perf_counter() - started

  print('>>> worming_up response, tokens, time:\n', resp, tokens, generation_time)

#########################################

def generate_one(
    msg: list[dict], 
    product_id: str,
    params={"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
) -> tuple[str, int]:
  messages_history = msg.copy()

  counter = 0
  errors = 0
  for attempt in range(1, MAX_ATTEMPTS + 1):
    counter += 1
    content, tokens = request_one(messages_history, params)

    error_dict = parse_and_validate(content, product_id)
    if not 'error' in error_dict:
      return content, tokens
    else:
      errors += 1
      print('>>> error: ', error_dict)
      print('>>> error content: ', content)

      if attempt == MAX_ATTEMPTS:
        return {"product_id": product_id, "error": "ResponseValidationError"}

      messages_history = messages_history + [
        {
          "role": "assistant",
          "content": content,
        },
        {
          "role": "user",
          "content": (
              "Предыдущий ответ не прошёл проверку. "
              f"Причина: {error_dict}. "
              "Исправь ответ и верни только JSON"
          ),
        }
      ]

##################################################
# file_name = './dev.jsonl'
file_name = './benchmark.jsonl'
with open(file_name, 'r', encoding='utf-8') as file:
  items = file.read().splitlines()
##################################################

def item_normalize(item: str) -> str:
  item_json = json.loads(item)
  reviews = item_json["reviews"][:MAX_REVIEWS]

  item_json["reviews"] = reviews
  return json.dumps(item_json, ensure_ascii=False)


def card_save(timestamp_str_dir, card, product_id):
  directory = Path(f"./outputs/{timestamp_str_dir}")
  directory.mkdir(parents=True, exist_ok=True)

  with open(f"{directory}/{product_id}.json", "w", encoding="utf-8") as file:
    file.write(card)


def params_save(timestamp_str_dir, params):
  directory = Path(f"./outputs/{timestamp_str_dir}")
  directory.mkdir(parents=True, exist_ok=True)

  with open(directory / "params.json", "w", encoding="utf-8") as file:
    json.dump(params, file, indent=4, ensure_ascii=False)
##################################################


from datetime import datetime
import time
import json

items_slice = items[:]
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
err = 0
count = 0

params = {
  "temperature":        0.3, 
  "top_p":              0.9, 
  "top_k":              50, 
  "repetition_penalty": 1.1, 
  "max_tokens":         350,
  "products":           len(items_slice), 
}

# worming_up()

started_batch = time.perf_counter()

for item in items_slice:
  count += 1

  item_norm = item_normalize(item)
  row_dict = {"role": "user", "content": item_norm}
  item_dict = json.loads(item)

  started = time.perf_counter()
  results, tokens = generate_one([row_dict], product_id=item_dict['product_id'], params=params)
  generation_time = time.perf_counter() - started

  if 'error' in results:
    err += 1
  else:
    card_save(timestamp_str, results, item_dict['product_id'])
    
  new_row = {"num": count, 'card': item_dict['product_id'], 'error': err, 'tokens': tokens, 'time_per_item': round(generation_time, 2)}
  print(new_row)

batch_generation_time = round(time.perf_counter() - started_batch, 2)
params_save(
  timestamp_str, 
  params | 
  {
    "errors": err, 
    "error_percent": round(100 * err / len(items_slice), 2), 
    "total_time_sec": batch_generation_time, 
    'time_per_item_sec': round(batch_generation_time / len(items_slice), 2)
  }
)


###############################################
files_path = Path("./outputs/cards-dev")
out = Path("./outputs/predictions.jsonl")

for file_path in files_path.rglob('P-*.json'):
  if file_path.is_file():
    content = file_path.read_text(encoding="utf-8")
    data = json.loads(content)

    with out.open("a", encoding="utf-8") as file:
      line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
      file.write(f'{line}\n')

files_path = Path("./outputs/cards-benchmark")
out = Path("./outputs/report.jsonl")

for file_path in files_path.rglob('P-*.json'):
  if file_path.is_file():
    content = file_path.read_text(encoding="utf-8")
    data = json.loads(content)

    with out.open("a", encoding="utf-8") as file:
      line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
      file.write(f'{line}\n')
