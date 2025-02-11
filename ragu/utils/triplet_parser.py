import re


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


