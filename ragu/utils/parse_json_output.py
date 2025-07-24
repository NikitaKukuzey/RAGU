import json
import re

from ragu.common.decorator import no_throw
from ragu.common.logger import logging


def check_quotes(text):
    lsts = text.split("\n")
    #print(lsts)
    for i in range(len(lsts)):
        if ("\"name\": " in lsts[i]) or ("\"description\": " in lsts[i]) or ("\"first_entity\": " in lsts[i]) or ("\"second_entity\": " in lsts[i]):
            n = lsts[i].count("\"")
            #print(n)
            if n > 4:
                ln = len(lsts[i])
                ind = lsts[i].find("\"", 0, ln)
                ind = lsts[i].find("\"", ind + 1, ln)
                ind = lsts[i].find("\"", ind + 1, ln)
                start = lsts[i][:ind + 1]
                end = lsts[i][ind + 1:]
                lsts[i] = start + end.replace("\"", "\'", n - 4)
    return "\n".join(lsts)

@no_throw
def extract_json(text: str):
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    text = text.replace("assistant", "")
    text = text.replace("\"\"", "\"")
    text = text.strip()
    text = text.replace('"enity_type"', '"entity_type"').replace('"desccription"', '"description"').replace('\\""', '\\"')
    text = check_quotes(text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    #print(match.group())
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print(f"Bad JSON")
    return None


def combine_report_text(report: dict):
    """
    Combine different sections of a report into a single text.

    :param report: The report contains title, summary, and findings.
    :return: A combined text representation of the report.
    """
    if not report:
        return ""

    parts = [
        report.get("title", ""),
        report.get("summary", ""),
    ]
    for finding in report.get("findings", []):
        parts.append(finding.get("summary", ""))
        parts.append(finding.get("explanation", ""))
    return " ".join(parts)


def create_text_from_community(report: dict):
    """
    Combine different sections of a report into a single text.

    :param report: The report contains title, summary, and findings.
    :return: A combined text representation of the report.
    """
    if not report:
        return ""

    community = report.get("community_report")
    entities = report.get("entities")
    relations = report.get("relations")

    parts = [
        community.get("title", ""),
        community.get("summary", ""),
    ]
    for finding in community.get("findings", []):
        parts.append(finding.get("summary", ""))
        parts.append(finding.get("explanation", ""))

    summaries = " ".join(parts)
    return f"Сущности: {entities}\n\nОтношения: {relations}\n\nСаммари: {summaries}"

