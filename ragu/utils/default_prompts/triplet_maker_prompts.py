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
   - **description**: Подробное описание сущности по приведенному тексту. Описание должно быть точным и максимально полным. 

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
            "description": "<описание сущности>"
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
            "description": "<описание сущности>"
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

delimiters = {"section_delimiter": "<SECTION>", "tuple_delimiter": "<|>", "record_delimiter": "##"}

original_like_prompt = """
**-Цель-**  
Дан текстовый документ и список типов сущностей. Необходимо:  
1. Выявить все сущности указанных типов в тексте.  
2. Определить все отношения между найденными сущностями.  

**-Шаги-**  
1. **Определение всех сущностей.**  
   Для каждой найденной сущности извлеките следующую информацию:  
   - **entity_name**: Название сущности с заглавной буквы.  
   - **entity_type**: Один из следующих типов: [{entity_types}].  
   - **description**: Полное описание характеристик и деятельности сущности.  
   Отформатируйте каждую сущность следующим образом:  
   ("entity_name"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<description>)

2. **Определение всех связанных пар (source_entity, target_entity) среди найденных сущностей.**  
   Для каждой пары связанных сущностей извлеките следующую информацию:  
   - **source_entity**: Название исходной сущности, найденное на шаге 1.  
   - **target_entity**: Название целевой сущности, найденное на шаге 1.  
   - **relationship_description**: Объяснение, почему указанные сущности связаны между собой.  
   - **relationship_strength**: Числовая оценка силы связи между сущностями.  
   Отформатируйте каждое отношение следующим образом:  
   ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. **Верни результат на английском языке** в виде единого списка всех сущностей и отношений, выявленных на шагах 1 и 2.  
   Используй **{record_delimiter}** в качестве разделителя списка.  
z
4. Разделяй блоки сущностей и связей разделителем **{section_delimiter}**. Поставь его в начале блока сущностей, после блока сущностей и после блока отношений

######################  
**-Примеры-**  
######################  

**Пример 1:**  

Типы сущностей: [person, technology, mission, organization, location]  
**Текст:**  
Пока Алекс сжимал челюсти, жужжание раздражения глушилось на фоне авторитарной уверенности Тейлор. Именно этот дух соперничества держал его в напряжении, ощущение того, что его и Джордана общая преданность открытиям была негласным бунтом против узкого видения контроля и порядка Круза.  
Затем Тейлор сделал нечто неожиданное. Он остановился рядом с Джорданом и на мгновение взглянул на устройство с чем-то похожим на благоговение.  
— Если эту технологию можно понять… — сказал Тейлор тихим голосом, — она может изменить правила игры для нас. Для всех нас.  
Ранее скрытая критика, казалось, ослабла, уступив место проблеску уважения к важности того, что находилось в их руках. Джордан поднял взгляд, и на мгновение их глаза встретились с Тейлором, молчаливая схватка воли смягчилась в тревожное перемирие.  

**Выходные данные:**
{section_delimiter}  
("entity_name"{tuple_delimiter}"Алекс"{tuple_delimiter}"person"{tuple_delimiter}"Алекс испытывает раздражение и наблюдает за динамикой между другими персонажами."){record_delimiter}  
("entity_name"{tuple_delimiter}"Тейлор"{tuple_delimiter}"person"{tuple_delimiter}"Тейлор демонстрирует авторитарную уверенность и проявляет благоговение перед устройством, что говорит о перемене его точки зрения."){record_delimiter}  
("entity_name"{tuple_delimiter}"Джордан"{tuple_delimiter}"person"{tuple_delimiter}"Джордан предан идее открытия и взаимодействует с Тейлором по поводу устройства."){record_delimiter}  
("entity_name"{tuple_delimiter}"Круз"{tuple_delimiter}"person"{tuple_delimiter}"Круз придерживается видения контроля и порядка, влияя на других персонажей."){record_delimiter}  
("entity_name"{tuple_delimiter}"Устройство"{tuple_delimiter}"technology"{tuple_delimiter}"Устройство является ключевым элементом истории, обладая потенциалом изменить правила игры."){record_delimiter}  
{section_delimiter}
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Тейлор"{tuple_delimiter}"Алекс подвержен влиянию авторитарной уверенности Тейлора и замечает изменения в его отношении к устройству."{tuple_delimiter}7){record_delimiter}  
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Джордан"{tuple_delimiter}"Алекс и Джордан разделяют стремление к открытиям, что контрастирует с видением Круза."{tuple_delimiter}6){record_delimiter}  
("relationship"{tuple_delimiter}"Тейлор"{tuple_delimiter}"Джордан"{tuple_delimiter}"Тейлор и Джордан взаимодействуют по поводу устройства, что приводит к моменту взаимного уважения и напряженного перемирия."{tuple_delimiter}8){record_delimiter}  
("relationship"{tuple_delimiter}"Джордан"{tuple_delimiter}"Круз"{tuple_delimiter}"Стремление Джордана к открытиям противостоит видению Круза о контроле и порядке."{tuple_delimiter}5){record_delimiter}  
("relationship"{tuple_delimiter}"Тейлор"{tuple_delimiter}"Устройство"{tuple_delimiter}"Тейлор проявляет благоговение перед устройством, подчеркивая его важность и потенциал."{tuple_delimiter}9)
{section_delimiter}

######################  
**-Реальные данные-**  
######################  

Типы сущностей: {entity_types}  
Текст:
"""

json_prompt = """
**-Цель-**  
Дан текстовый документ и список типов сущностей, связей. Необходимо выявить все сущности указанных типов в тексте, а также все связи между найденными сущностями.  

**-Шаги-**  
1. **Идентификация всех сущностей.**  
   Для каждой найденной сущности извлечь следующую информацию:  
   - **name**: Нормализованное название сущности с заглавной буквы. 
    Под нормализацией подразумевается приведение слова к начальной форме. 
    Пример: рождеству -> Рождество, кошек -> Кошки, Павла -> Павел.
   - **entity_type**: Тип сущности. Допустимые типы: AGE AWARD CITY COUNTRY CRIME DATE DISEASE DISTRICT EVENT FACILITY FAMILY IDEOLOGY LANGUAGE LAW LOCATION MONEY NATIONALITY NUMBER ORDINAL ORGANIZATION PENALTY PERCENT PERSON PRODUCT PROFESSION RELIGION STATE_OR_PROVINCE TIME WORK_OF_ART
   - **description**: Подробное описание сущности по приведенному тексту. Описание должно быть точным и максимально полным. 

2. **Определение связей между сущностями.**  
   На основе сущностей, найденных на первом шаге, определить все пары (**source_entity**, **target_entity**), которые *явно связаны* между собой.  
   Для каждой такой пары извлечь следующую информацию:
   - **first_entity**: Название исходной сущности (как определено в шаге 1)  
   - **second_entity**: Название целевой сущности (как определено в шаге 1)
   - **rel_type**: Тип отношения между двумя сущностями. Допустимые типы: ABBREVIATION KNOWS AGE_IS AGE_DIED_AT ALTERNATIVE_NAME AWARDED_WITH PLACE_OF_BIRTH CAUSE_OF_DEATH DATE_DEFUNCT_IN DATE_FOUNDED_IN DATE_OF_BIRTH DATE_OF_CREATION DATE_OF_DEATH POINT_IN_TIME PLACE_OF_DEATH FOUNDED_BY HEADQUARTERED_IN IDEOLOGY_OF LOCATED_IN SPOUSE MEDICAL_CONDITION MEMBER_OF ORGANIZES ORIGINS_FROM OWNER_OF PARENT_OF PLACE_RESIDES_IN PRICE_OF PRODUCES RELATIVE RELIGION_OF SCHOOLS_ATTENDED SIBLING SUBEVENT_OF SUBORDINATE_OF TAKES_PLACE_IN WORKPLACE WORKS_AS START_TIME END_TIME CONVICTED_OF PENALIZED_AS PART_OF HAS_CAUSE AGENT PARTICIPANT_IN INANIMATE_INVOLVED EXPENDITURE INCOME
   - **description**: Описание связи между двумя сущностями. 
   - **strength**: Числовой показатель, отражающий силу связи между сущностями в диапазоне от 0 до 10, где 0 - слабая связь, 10 - сильная связь.

3. **Вывести результат на русском языке** СТРОГО в следующем виде, без добавочных символов(кавычек, обозначение и т.д.):
{{
    "entities": [
        {{
            "name": "<название сущности>",
            "entity_type": "<тип сущности>",
            "description": "<описание сущности>"
        }}
    ],
    "relations": [
        {{
            "first_entity": "<название первой сущности>",
            "second_entity": "<название второй сущности>",
            "rel_type": "<тип отношения между сущностями>",
            "description": "<описание связи>",
            "strength": "<сила связи>"
        }}
    ]
}}
4. **Проверь ответ на формат json**, содержит только данные в формате json.

Текст:
{{text}}
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