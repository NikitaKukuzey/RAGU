prompt = """
**-Цель-**  
Дан текстовый документ и список типов сущностей. Необходимо выявить все сущности указанных типов в тексте, а также все связи между найденными сущностями.  

**-Шаги-**  
1. **Идентификация всех сущностей.**  
   Для каждой найденной сущности извлечь следующую информацию:  
   - **entity_name**: Нормализованное название сущности с заглавной буквы. 
    Под нормализацией подразумевается приведение слова к начальной форме. 
    Пример: рождеству -> Рождество, кошек -> Кошки, Павла -> Павел.
   - **entity_type**: Тип сущности. Допустимые типы: {allowed_entity_types}
   - **entity_description**: Подробное описание характеристик и деятельности сущности  

2. **Определение связей между сущностями.**  
   На основе сущностей, найденных на первом шаге, определить все пары (**source_entity**, **target_entity**), которые *явно связаны* между собой.  
   Для каждой такой пары извлечь следующую информацию:  
   - **source_entity**: Название исходной сущности (как определено в шаге 1)  
   - **target_entity**: Название целевой сущности (как определено в шаге 1)  
   - **relationship_description**: Описание связи между двумя сущностями. 
   - **relationship_strength**: Числовой показатель, отражающий силу связи между сущностями в диапазоне от 0 до 5, где 0 - слабая связь, 5 - сильная связь.

3. **Вывести результат на русском языке** строго в следующем виде:
{{
    "entities": [
        {{
            "entity_name": "<название сущности>",
            "entity_type": "<тип сущности>",
            "entity_description": "<описание сущности>"
        }}
    ],
    "relationships": [
        {{
            "source_entity": "<название исходной сущности>",
            "target_entity": "<название целевой сущности>",
            "relationship_description": "<описание связи>",
            "relationship_strength": <число от 0 до 5>
        }}
    ]
}}

Текст:
"""

validation_prompt = """
**-Цель-**  
Ты - помощник, который проверяет правильность введенных сущностей и отношений.  
Тебе на вход подается список сущностей, список отношений и текст, где они были выделены.  
Твоя задача - проверить, все ли сущности были выделены из текста и имеют правильный тип и описание и вернуть полный список всех сущностей и отношений.  

**-Шаги-**  
1. **Проверка сущностей.**  
   - Если сущность была пропущена, добавь её в список сущностей с описанием и её типом.  
   - **Список допустимых типов сущностей:** {allowed_entity_types}.  
   - Верни список всех сущностей: и ранее выделенных, и тех, что были пропущены.

2. **Проверка отношений.**  
   - Если отношение было пропущено, добавь его в список отношений с описанием и силой связи.  
   - Верни список всех отношений: и ранее выделенных, и тех, что были пропущены. 

3. **Вывести результат на русском языке** строго в следующем виде:
{{
    "entities": [
        {{
            "entity_name": "<название сущности>",
            "entity_type": "<тип сущности>",
            "entity_description": "<описание сущности>"
        }}
    ],
    "relationships": [
        {{
            "source_entity": "<название исходной сущности>",
            "target_entity": "<название целевой сущности>",
            "relationship_description": "<описание связи>",
            "relationship_strength": <число от 0 до 5>
        }}
    ]
}}
"""

def _generate_prompt(input_prompt, **kwargs):
    return input_prompt.format(**kwargs)


default_entities = [
    "ОРГАНИЗАЦИЯ",
    "ПЕРСОНА",
    "МЕСТОПОЛОЖЕНИЕ",
    "СОБЫТИЕ"
]
nerel_entities = [
    "ВОЗРАСТ",
    "СЕМЬЯ",
    "НАГРАДА",
    'ИДЕОЛОГИЯ',
    'ПРОЦЕНТ',
    'ГОРОД',
    'ЯЗЫК',
    'ЧЕЛОВЕК',
    'СТРАНА',
    'ЗАКОН',
    'ПРОДУКТ',
    'ПРЕСТУПЛЕНИЕ',
    'МЕСТОПОЛОЖЕНИЕ',
    'ПРОФЕССИЯ',
    'ДАТА',
    'ДЕНЬГИ',
    'РЕЛИГИЯ',
    'БОЛЕЗНЬ',
    'НАЦИОНАЛЬНОСТЬ',
    'ОБЛАСТЬ ИЛИ КРАЙ',
    'РАЙОН',
    'НОМЕР ДОМА',
    'ВРЕМЯ',
    'СОБЫТИЕ',
    'ПОРЯДКОВЫЙ НОМЕР',
    'ПРОИЗВЕДЕНИЕ ИСКУССТВА',
    'ОБЪЕКТ',
    'ОРГАНИЗАЦИЯ']

english_default_entities = [
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "EVENT"
]

english_nerel_entities = [
    "AGE",
    "FAMILY",
    "AWARD",
    'IDEOLOGY',
    'PERCENT',
    'CITY',
    'LANGUAGE',
    'PERSON',
    'COUNTRY',
    'LAW',
    'PRODUCT',
    'CRIME',
    'PENALTY',
    'PROFESSION',
    'DATE',
    'MONEY',
    'RELIGION',
    'DISEASE',
    'NATIONALITY',
    'STATE_OR_PROV',
    'ORDINAL',
    'TIME',
    'EVENT',
    'DISTRICT',
    'WORK_OF_ART',
    'ORGANIZATION',
    'FACILITY',
    'NUMBER',
    'LOCATION',
]

# PROMPTS
prompts = {
    'default':  _generate_prompt(prompt, allowed_entity_types=", ".join(default_entities)),
    'nerel': _generate_prompt(prompt, allowed_entity_types=", ".join(nerel_entities))
}

validation_prompts = {
    'default':  _generate_prompt(validation_prompt, allowed_entity_types=", ".join(default_entities)),
    'nerel': _generate_prompt(validation_prompt, allowed_entity_types=", ".join(nerel_entities))
}

english_entities_dict = {
    'default': english_default_entities,
    'nerel': english_nerel_entities,
}