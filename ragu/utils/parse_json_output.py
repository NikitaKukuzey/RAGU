import re
import json


from ragu.common.decorator import no_throw


@no_throw
def extract_json(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def combine_report_text(report: dict):
    """
    Combine different sections of a report into a single text.

    :param report: The report containing title, summary, and findings.
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

