import re

import pandas as pd


def parse_relations(text):
    pattern = r'\d+\.\s*(.+?)\s*<\|>\s*(.+?)\s*<\|>\s*(.+?)\s*<\|>\s*(.+?)\s*<\|>\s*(.+?)\s*<\|>\s*(.+?)\s*$'
    matches = re.findall(pattern, text, re.MULTILINE)
    matches = [(entity1.strip().lower(),
                entity1_type.strip().lower(),
                relation.strip().lower(),
                relation_type.strip().lower(),
                entity2.strip().lower(),
                entity2_type.strip().lower())
               for entity1, entity1_type, relation, relation_type, entity2, entity2_type in matches]
    return matches


def parse_description(text):
    pattern = r"\d+\.\s*(.*)\s*<\|>\s*(.*)"
    matches = re.findall(pattern, text, re.MULTILINE)
    matches = [
        {'Entity': entity.strip().lower(),
         'Description': description.strip().lower()}
        for entity, description in matches]

    return matches

def parse_llm_response(response: str):
    sections = re.split(r'<\|\|>', response)

    if len(sections) < 3:
        raise ValueError("Некорректный формат ответа модели")

    entities_section = sections[1].strip()
    relationships_section = sections[2].strip()

    entities = []
    relationships = []

    entity_lines = entities_section.split('\n')
    for line in entity_lines:
        match = re.match(r'(.+?) <\|> (.+?) <\|> (.+)', line)
        if match:
            entity_name, entity_type, entity_description = match.groups()
            entities.append({
                "entity_name": entity_name.strip(),
                "entity_type": entity_type.strip(),
                "entity_description": entity_description.strip()
            })

    relationship_lines = relationships_section.split('\n')
    for line in relationship_lines:
        match = re.match(r'(.+?) <\|> (.+?) <\|> (.+?) <\|> (\d+)', line)
        if match:
            source_entity, target_entity, relationship_description, relationship_strength = match.groups()
            relationships.append({
                "source_entity": source_entity.strip(),
                "target_entity": target_entity.strip(),
                "relationship_description": relationship_description.strip(),
                "relationship_strength": int(relationship_strength.strip())
            })

    return pd.DataFrame(entities), pd.DataFrame(relationships)


