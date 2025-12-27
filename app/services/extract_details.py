import re
from typing import Dict, List

date_pattern = re.compile(r"(\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)", flags=re.I)
party_pattern = re.compile(r"(party\s+[A-Z]|\bparty\s+[A-Z][a-zA-Z0-9_]*\b|between\s+([A-Z][\w\s,]+)\s+and\s+([A-Z][\w\s,]+))", flags=re.I)

def extract_important_details(text: str) -> Dict[str, List[str]]:
    text = text or ""
    dates = list({m.group(0).strip() for m in date_pattern.finditer(text)})

    parties = []
    for m in party_pattern.finditer(text):
        parties.append(m.group(0).strip())

    obligations = []
    for s in re.split(r"[.\n]", text):
        if re.search(r"\b(shall|must|will|agree|agrees|obligat)\b", s, flags=re.I):
            obligations.append(s.strip())

    termination = [s.strip() for s in re.split(r"[.\n]", text) if re.search(r"\b(terminate|termination|terminate this agreement|termination may)\b", s, flags=re.I)]

    penalties = [s.strip() for s in re.split(r"[.\n]", text) if re.search(r"\b(penalti|penalty|fine|liquidated damages)\b", s, flags=re.I)]

    return {
        "parties": parties,
        "dates": dates,
        "obligations": obligations,
        "termination": termination,
        "penalties": penalties,
        "risks": []
    }

