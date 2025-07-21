import json
import re

from ragu.common.decorator import no_throw
from ragu.common.logger import logging


@no_throw
def extract_json(text: str):
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logging.warning(f"Bad JSON: {text}")
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

