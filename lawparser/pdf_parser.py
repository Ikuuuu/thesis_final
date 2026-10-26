"""국가법령정보센터 법령 PDF → 구조화 문서(LawDocument) 파서.

처리 단계
  1. 파일명 파싱   : "제목(법령종류)(제N호)(시행일).pdf" → 제목·종류·공포번호·시행일·발령기관
  2. 본문 메타 파싱 : [시행 ...] [법령종류 제N호, 공포일, 제·개정구분], 약칭, 소관부처
  3. 본문/부칙 분리 : "부 칙" 헤더 기준
  4. 조문 분할      : "제N조(제목)" / "제N조의M" 헤더 기준 → {no, title, text}

법령종류 문자열은 OWL 클래스명(Law, EnforcementOrdinance, AdministrativeOrder …)으로 정규화한다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("kge.data.pdf_parser")


DOC_TYPE_MAP = {
    "법률":             "Law",
    "대통령령":          "EnforcementOrdinance",
    "총리령":            "EnforcementRule",
    "국방부령":          "EnforcementRule",
    "부령":              "EnforcementRule",
    "특례규정":          "SpecialRegulation",
    "국방부훈령":         "AdministrativeOrder",
    "방위사업청훈령":      "AdministrativeOrder",
    "훈령":              "AdministrativeOrder",
    "공동훈령":          "JointAdministrativeOrder",
    "예규":              "AdministrativeRule",
    "고시":              "AdministrativeNotice",
}

@dataclass
class LawDocument:
    law_id: str
    title: str
    short_title: Optional[str]
    doc_type: str
    doc_type_raw: str
    issuing_org: Optional[str]
    promulgation_no: Optional[str]
    promulgation_date: Optional[str]
    effective_date: Optional[str]
    revision_type: Optional[str]
    jurisdiction: List[str] = field(default_factory=list)
    articles: List[Dict[str, Any]] = field(default_factory=list)
    addenda: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


FILENAME_RE = re.compile(
    r"^(?P<title>.+?)\((?P<dtype>[^)]+)\)\((?P<pno>제[0-9]+호)\)\((?P<edate>[0-9]{8})\)\.pdf$"
)


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    m = FILENAME_RE.match(filename)
    if not m:
        return {"title": Path(filename).stem, "doc_type_raw": None,
                "promulgation_no": None, "effective_date": None, "issuing_org": None}
    dtype_raw = m.group("dtype").strip()
    issuing_org = None
    for org in ["국방부", "방위사업청", "산업통상자원부", "총리실"]:
        if dtype_raw.startswith(org):
            issuing_org = org
            break
    if dtype_raw == "공동훈령":
        issuing_org = "공동"
    return {
        "title":           m.group("title").strip(),
        "doc_type_raw":    dtype_raw,
        "promulgation_no": m.group("pno"),
        "effective_date":  m.group("edate"),
        "issuing_org":     issuing_org,
    }


def normalize_doc_type(dtype_raw: str) -> str:
    if not dtype_raw:
        return "LegalDocument"
    keys_sorted = sorted(DOC_TYPE_MAP.keys(), key=len, reverse=True)
    for k in keys_sorted:
        if dtype_raw == k or dtype_raw.endswith(k):
            return DOC_TYPE_MAP[k]
    return "LegalDocument"


META_RE = re.compile(
    r"\[\s*시행\s+([0-9.\s]+?)\s*\.\s*\]\s*"
    r"\[\s*(?P<dtype>[^\s\d]+)\s+(?P<pno>제[0-9]+호)\s*,\s*"
    r"([0-9.\s]+?)\s*\.\s*,\s*(?P<rtype>[^]]+)\]"
)
SHORT_TITLE_RE = re.compile(r"약칭\s*:\s*([^)\n]+)")
JURISDICTION_RE = re.compile(
    r"^(?P<org>[가-힣]+(?:부|청|위원회|원|처|실|총리실|위원장))\s*\("
)


def _normalize_date(s: str) -> Optional[str]:
    if not s:
        return None
    parts = re.findall(r"\d+", s)
    if len(parts) >= 3:
        y, m, d = parts[0], parts[1].zfill(2), parts[2].zfill(2)
        return f"{y}{m}{d}"
    return None


def parse_metadata_block(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "short_title": None, "effective_date_meta": None,
        "promulgation_date": None, "doc_type_meta": None,
        "promulgation_no_meta": None, "revision_type": None,
        "jurisdiction": [],
    }
    head = text[:3000]
    m = SHORT_TITLE_RE.search(head)
    if m:
        meta["short_title"] = m.group(1).strip()
    m = META_RE.search(head)
    if m:
        meta["effective_date_meta"] = _normalize_date(m.group(1))
        meta["doc_type_meta"] = m.group("dtype")
        meta["promulgation_no_meta"] = m.group("pno")
        # 주의: group(3) 은 이름 있는 그룹(pno)이므로 공포일은 4번째 그룹이다.
        #       (v1.0 에서는 group(3) 을 써서 공포일이 항상 None 이던 버그를 수정)
        meta["promulgation_date"] = _normalize_date(m.group(4))
        meta["revision_type"] = m.group("rtype").strip()
    seen = set()
    for line in head.splitlines():
        line = line.strip()
        m = JURISDICTION_RE.match(line)
        if m:
            org = m.group("org").strip()
            if org and org not in seen and len(org) <= 12:
                seen.add(org)
                meta["jurisdiction"].append(org)
    return meta


ADDENDUM_HEADER_RE = re.compile(r"^\s*부\s*칙\s*(?:\(.*?\))?\s*$", re.MULTILINE)


def split_main_and_addenda(text: str) -> tuple[str, str]:
    matches = list(ADDENDUM_HEADER_RE.finditer(text))
    if not matches:
        return text, ""
    first = matches[0]
    return text[: first.start()], text[first.start():]


def parse_articles(text: str, legacy: bool = False) -> List[Dict[str, Any]]:
    """조문 헤더 "제N조(제목)" 기준으로 본문을 분할한다.

    legacy=False (기본, v1.1) : 헤더와 같은 줄에 이어지는 본문도 조문 텍스트에 포함한다.
                                "제2조의2" 형식의 가지조문도 별도 조문으로 인식한다.
    legacy=True  (v1.0)       : 졸업논문 데이터셋 v1.0 재현용. 두 가지 알려진 문제가 있다.
                                ① 헤더 줄 전체를 버려 첫 줄의 「」 인용이 누락됨
                                ② "제N조의M" 가지조문을 인식하지 못해 해당 조문 전체가 누락됨
    """
    if legacy:
        # v1.0 : "제2조의2" 형식을 인식하지 못해 가지조문(조의N)이 앞 조문 번호와 중복 처리되어 누락됨
        header_re = r"^제(\d+(?:의\d+)?)\s*조(?:\s*\(([^)]*)\))?"
    else:
        # v1.1 : 법령 표기 "제2조의2(제목)" 을 올바르게 인식 → no = "2의2"
        header_re = r"^제(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*\(([^)]*)\))?"
    headers = list(re.finditer(header_re, text, re.MULTILINE))
    articles: List[Dict[str, Any]] = []
    seen_no = set()
    for i, h in enumerate(headers):
        if legacy:
            no, title = h.group(1), h.group(2)
        else:
            no = h.group(1) + (f"의{h.group(2)}" if h.group(2) else "")
            title = h.group(3)
        if no in seen_no:
            continue
        seen_no.add(no)
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        if legacy:
            body_lines = body.split("\n", 1)
            full_body = body_lines[1].strip() if len(body_lines) > 1 else ""
        else:
            full_body = text[h.end():end].strip()
        articles.append({
            "no": no,
            "title": (title or "").strip(),
            "text": full_body,
        })
    return articles


def read_pdf_text(pdf_path: str | Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf 설치 필요 : pip install pypdf") from e
    reader = PdfReader(str(pdf_path))
    chunks: List[str] = []
    for p in reader.pages:
        try:
            chunks.append(p.extract_text() or "")
        except Exception as e:
            logger.warning(f"page 추출 실패 ({Path(pdf_path).name}): {e}")
    return "\n".join(chunks)


def parse_pdf(pdf_path: str | Path, legacy: bool = False) -> LawDocument:
    pdf_path = Path(pdf_path)
    fn_meta = parse_filename(pdf_path.name)
    raw_text = read_pdf_text(pdf_path)
    body_meta = parse_metadata_block(raw_text)
    doc_type_raw = fn_meta.get("doc_type_raw") or body_meta.get("doc_type_meta") or ""
    doc_type = normalize_doc_type(doc_type_raw)
    title = fn_meta.get("title") or pdf_path.stem
    main_text, addendum_text = split_main_and_addenda(raw_text)
    return LawDocument(
        law_id=pdf_path.stem,
        title=title,
        short_title=body_meta.get("short_title"),
        doc_type=doc_type,
        doc_type_raw=doc_type_raw,
        issuing_org=fn_meta.get("issuing_org"),
        promulgation_no=fn_meta.get("promulgation_no") or body_meta.get("promulgation_no_meta"),
        promulgation_date=body_meta.get("promulgation_date"),
        effective_date=fn_meta.get("effective_date") or body_meta.get("effective_date_meta"),
        revision_type=body_meta.get("revision_type"),
        jurisdiction=body_meta.get("jurisdiction", []),
        articles=parse_articles(main_text, legacy=legacy),
        addenda=parse_articles(addendum_text, legacy=legacy) if addendum_text else [],
        raw_text=raw_text,
    )


def parse_directory(pdf_dir: str | Path, legacy: bool = False) -> List[LawDocument]:
    pdf_dir = Path(pdf_dir)
    documents: List[LawDocument] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            doc = parse_pdf(pdf_path, legacy=legacy)
            documents.append(doc)
        except Exception as e:
            logger.error(f"  ✗ {pdf_path.name} 실패 : {e}")
    return documents
