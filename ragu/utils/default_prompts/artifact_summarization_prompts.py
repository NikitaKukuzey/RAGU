from collections import defaultdict

artifacts_summarization_prompt = defaultdict()

artifacts_summarization_prompt["summarize_entity_descriptions"] = """
**Задача:** по данной сущности и набору фраз для её описания сгенерировать одно общее краткое описание. Оно должно быть лаконичным и непротиворечивым.

Верни ответ строго в формате json следующей структуры: 
'''
{{
    "description": "суммаризированное описание"
}}
'''

Данные:
{input_text}
"""

artifacts_summarization_prompt["summarize_relation_descriptions"] = """
**Задача:** по приведённым парам сущностей и набору фраз для описания их отношений сгенерировать одно общее краткое описание отношений между этими сущностями.
Описание должно быть лаконичным и непротиворечивым.

Верни ответ строго в формате json следующей структуры: 
'''
{{
    "description": "суммаризированное описание"
}}
'''

Данные:
{input_text}
"""