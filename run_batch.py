"""
Usage:
python3 ./run_batch.py -i <file>.jsonl

"""
import argparse
import ast
import json
import time
from datetime import datetime
from pathlib import Path
from ollama import Client
from check_metrics import CardRequest, ResponseValidationError, parse_and_validate, item_normalize

MAX_ATTEMPTS = 3 
MAX_TOKENS = 1500

MODEL='llama3.1:8b'
client = Client(host='http://localhost:11434')


def request_one(
    msg: list[dict], 
    params={"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350}
) -> dict:
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
      model=MODEL,  
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

  return {
    "content": content,
    "input_tokens": response.prompt_eval_count,
    "output_tokens": response.eval_count,
  }


def worming_up():
  print('>>> worming up')
  item = {"product_id": "P-0026", "title": "Смартфон Grafo G9", "category": "Электроника / Смартфоны", "attributes": {"бренд": "Grafo", "экран": "6.7\"", "память": "256 ГБ", "оперативная память": "8 ГБ", "аккумулятор": "6000 мА·ч", "основная камера": "50 Мп"}, "reviews": ["Пользуюсь пару недель. Камера снимает достойно даже вечером", "Брал для себя. Звонки чистые, собеседника слышно хорошо", "В целом Быстро заряжается, за час почти полный", "Заказывал на подарок. Работает шустро, приложения не тормозят", "Брал для себя. Камера мылит фото при недостатке света", "Пользуюсь пару недель. К вечеру заряд просаживается быстрее ожидаемого", "Пользуюсь пару недель. Экран яркий, на солнце всё видно", "Пользуюсь пару недель. Экран яркий, на солнце всё видно", "Пользуюсь каждый день. Экран яркий, на солнце всё видно", "Быстро заряжается, за час почти полный", "В руке лежит удобно, не скользит", "Динамик тихий, в шумном месте плохо слышно", "Работает шустро, приложения не тормозят", "В руке лежит удобно, не скользит", "Коротко: Работает шустро, приложения не тормозят", "Купил на замену старому. Обычный телефон на каждый день, без сюрпризов", "Брал для себя. Батарея держит полтора дня без подзарядки", "В целом Работает шустро, приложения не тормозят"]}

  row_dict = [{"role": "user", "content": f"""{item}"""}]

  started = time.perf_counter()
  resp = request_one(row_dict)
  generation_time = time.perf_counter() - started

  print('>>> worming up (sek): ', round(generation_time, 2))


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
    resp = request_one(messages_history, params)

    error_dict = parse_and_validate(resp['content'], product_id)
    if not 'error' in error_dict:
      return resp | {'attempts': attempt}
    else:
      errors += 1
      print('>>> error: ', error_dict)
      print('>>> error content: ', resp['content'])

      if attempt == MAX_ATTEMPTS:
        return {"product_id": product_id, "error": "ResponseValidationError"}

      messages_history = messages_history + [
        {
          "role": "assistant",
          "content": resp['content'],
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Card generation script")
    parser.add_argument("--input", "-i", type=str, required=True, help="input file")
    return parser.parse_args()


##################################################
def main():
  print('>>> starting card generation')

  args = parse_arguments()
  with open(f'./{args.input}', 'r', encoding='utf-8') as file:
    items = file.read().splitlines()

  items_slice = items[:]
  timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
  err = 0
  count = 0

  input_tokens = 0
  output_tokens = 0
  attempts = 0

  out = Path(f"./outputs/predictions_{timestamp_str}.jsonl")
  out_metrics = Path(f"./outputs/predictions_metrics_{timestamp_str}.jsonl")

  params = {
    "temperature":        0.3, 
    "top_p":              0.9, 
    "top_k":              50, 
    "repetition_penalty": 1.1, 
    "max_tokens":         350,
    "products":           len(items_slice), 
  }

  worming_up()

  started_batch = time.perf_counter()

  for item in items_slice:
    count += 1

    item_norm = item_normalize(item)
    row_dict = {"role": "user", "content": item_norm}
    item_dict = json.loads(item)

    started = time.perf_counter()
    results = generate_one([row_dict], product_id=item_dict['product_id'], params=params)
    generation_time = time.perf_counter() - started

    if 'error' in results:
      err += 1
    else:
      input_tokens += results['input_tokens']
      output_tokens += results['output_tokens']
      attempts += results['attempts']

      with out.open("a", encoding="utf-8") as file:
        data = ast.literal_eval(results["content"])
        file.write(f'{json.dumps(data, ensure_ascii=False)}\n')

    new_row = {
      "num": count, 
      'card': item_dict['product_id'], 
      'error': err, 
      'input_tokens': results['input_tokens'], 
      'output_tokens': results['output_tokens'], 
      'attempts': results['attempts'], 
      'time_per_item': round(generation_time, 2)
    }
    print(new_row)

  batch_generation_time = round(time.perf_counter() - started_batch, 2)

  metrics = params | {
    "errors": err, 
    "error_percent": round(100 * err / len(items_slice), 2), 
    'input_tokens': input_tokens,
    'output_tokens': output_tokens,
    'attempts': attempts,
    "total_time_sec": batch_generation_time, 
    'time_per_item_sec': round(batch_generation_time / len(items_slice), 2)
  }
  print(metrics)
  with out_metrics.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
