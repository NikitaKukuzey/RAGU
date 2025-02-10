import re


def parse_relations(text):
    pattern = r'\d+\.\s*(.+?)\s*<\|>\s*(.+?)\s*<\|>\s*(.+?)\s*$'
    matches = re.findall(pattern, text, re.MULTILINE)
    matches = [(entity1.lower(), relation.lower(), entity2.lower()) for entity1, relation, entity2 in matches]
    return matches

def parse_description(text):
    pattern = r"\d+\.\s*(.*)\s*<\|>\s*(.*)"
    matches = re.findall(pattern, text, re.MULTILINE)
    matches = [{'entity': entity.lower(), 'description': description.lower()} for entity, description in matches]

    return matches


