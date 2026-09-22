"""선정된 후보 논문 PDF -> 섹션 기반 청크 분할 -> 임베딩 -> FAISS 벡터스토어.

청킹 전략 (기술조사 에이전트의 오귀속/누락 문제를 줄이기 위한 설계):
- 페이지마다 반복되는 러닝 헤더·페이지 번호·arXiv 워터마크를 제거한다.
- 섹션 헤더("3.1. Title" / "3.1 Title" / Abstract / References / Appendix A ...)를 기준으로
  먼저 나누고, 섹션 안에서만 문자 단위 분할을 한다. 그래서 Abstract·Conclusion처럼 짧은
  섹션도 다른 섹션에 묻히지 않고 독립 청크가 된다.
- "Table N:" 캡션+숫자 행, "Algorithm N:" 의사코드 블록은 쪼개지 않고 한 청크로 보호한다.
- References·Acknowledgments는 어떤 관점 평가에도 근거가 되지 않으므로 인덱싱하지 않는다.
- 각 청크 metadata: section / section_num / section_kind / block_type / chunk_id / page.
  임베딩 텍스트 앞에는 "[§섹션]" 한 줄을 붙여 섹션 어휘가 검색에 반영되게 한다.

총 200페이지 한도(과제 조건)를 넘지 않도록, 로딩 시 페이지 수를 세어 경고한다.
PDF가 data/ 에 없으면 예외 대신 warning을 남기고 None을 반환해 그래프가 계속
진행될 수 있게 한다 (기술 조사 에이전트가 summary 필드로 폴백).
"""
import bisect
import os
import re
from collections import Counter
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_embeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
MAX_TOTAL_PAGES = 200

# 청킹 방식이 바뀌면 올린다. 구 인덱스(section 메타데이터 없음)를 조용히 재사용하지 않도록
# 저장 경로에 포함된다.
INDEX_VERSION = "v2"

# bge-m3는 최대 8192 토큰(약 5000~6000자)까지 지원하므로, 문단 단위 맥락이
# 끊기지 않도록 기존 e5-small 기준(800자)보다 넉넉한 청크 크기를 사용한다.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
# 표/알고리즘 블록은 원칙적으로 쪼개지 않지만, 이 길이를 넘으면 임베딩 한도를 위해 나눈다.
MAX_BLOCK_CHARS = 6000
# 섹션 안에서 이보다 짧은 산문 청크는 바로 앞 산문 청크에 합친다 (표 사이에 낀 한두 줄 방지).
MIN_PROSE_CHARS = 300

SKIP_SECTION_KINDS = {"references", "acknowledgments", "front_matter"}

# --- 헤더 패턴 -------------------------------------------------------------
# ICML: "3.1. Title" / ACM: "3.1 Title" / pypdf가 공백을 떨어뜨린 "3.3.KIVI: ..." 까지 허용
_NUMBERED_HEADER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*([A-Z][^\n]{2,80})$")
_SPECIAL_HEADER = re.compile(r"^(Abstract|References|Bibliography|Acknowledg\w*|Impact Statement)\b")
# 부록 헤더("A. Detailed Implementations")는 References 이후에만 인정한다 ("A large ..." 오탐 방지)
_APPENDIX_HEADER = re.compile(r"^(?:Appendix\s+)?([A-H](?:\.\d+)*)\.?\s+([A-Z][A-Za-z\-:,&\s]{3,60})$")

# "2 GB of memory ..." 같은 수치 문장이 섹션 2로 오인되지 않도록 단위로 시작하는 제목은 거른다
_UNIT_TITLE = re.compile(r"^(GB|MB|KB|TB|B|GHz|MHz|ms|us|µs|s|x|×|%|bit|bits|GB/s|MB/s)\b")
# 참고문헌 항목 판별 (부록 헤더 뒤에 이런 줄이 이어지면 논문 제목 줄이 헤더로 오인된 것).
# 부록 본문에도 "(Author, 2023)" 인용이 흔하므로 연도는 줄 끝에 있을 때만 서지로 본다.
_BIB_LINE = re.compile(r"arXiv|Proceedings|pp\.|vol\.|doi|preprint|\b(19|20)\d{2}\.?$", re.I)

_TABLE_START = re.compile(r"^Table\s*\d+\s*[:.]")
_ALGO_START = re.compile(r"^Algorithm\s*\d+\s*[:.]")
_FIGURE_START = re.compile(r"^Figure\s*\d+\s*[:.]")
_ALGO_KEYWORD = re.compile(
    r"^(procedure|function|input|output|return|end|if|else|for|while|parameter|require|ensure|\d+:)",
    re.I,
)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_ARXIV_MARK = re.compile(r"^arXiv:\d{4}\.\d{4,5}")


def _load_pdf(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, f"[RAG] data/{filename} 파일을 찾을 수 없어 원문 검색 없이 진행합니다."
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages, None


# --- 1) 페이지 정리 -----------------------------------------------------------
def _clean_pages(pages: list[Document]) -> list[list[tuple[str, int]]]:
    """페이지별 (line, page) 목록. 러닝 헤더·페이지 번호·arXiv 워터마크를 제거한다."""
    first_lines = Counter()
    for p in pages:
        stripped = p.page_content.strip()
        if stripped:
            first_lines[stripped.split("\n", 1)[0].strip()] += 1
    # 과반 페이지에서 첫 줄이 같으면 러닝 헤더로 본다 (KIVI는 매 페이지 상단에 논문 제목이 반복됨)
    running = {line for line, n in first_lines.items() if n >= max(2, len(pages) // 2)}

    cleaned = []
    for i, p in enumerate(pages):
        page_no = p.metadata.get("page", i)
        lines = [l.rstrip() for l in p.page_content.split("\n")]
        while lines and (not lines[0].strip() or lines[0].strip() in running):
            lines.pop(0)
        while lines and (not lines[-1].strip() or _PAGE_NUMBER.match(lines[-1].strip())):
            lines.pop()
        cleaned.append([(l, page_no) for l in lines if l.strip() and not _ARXIV_MARK.match(l.strip())])
    return cleaned


# --- 2) 섹션 분할 -------------------------------------------------------------
def _section_kind(label: str) -> str:
    low = label.lower()
    if low.startswith("abstract"):
        return "abstract"
    if low.startswith(("references", "bibliography")):
        return "references"
    if low.startswith("acknowledg"):
        return "acknowledgments"
    if low.startswith("appendix"):
        return "appendix"
    title = re.sub(r"^\d+(\.\d+)*\s*", "", low)
    if "related work" in title or "related studies" in title or "prior work" in title:
        return "related_work"
    if any(w in title for w in ("conclusion", "future work", "discussion")):
        return "conclusion"
    if "introduction" in title:
        return "introduction"
    if any(w in title for w in ("background", "motivation", "preliminar")):
        return "background"
    # "Evaluation Methodology"처럼 두 종류 어휘가 섞이면 실험으로 본다 (method보다 먼저 검사)
    if any(w in title for w in ("experiment", "evaluation", "result", "ablation", "comparison",
                                "efficiency", "accuracy", "performance", "setting")):
        return "experiments"
    if any(w in title for w in ("method", "design", "architecture", "implementation", "algorithm",
                                "approach", "system", "interface", "management", "scheduling")):
        return "method"
    return "other"


def _num_key(num: str) -> tuple:
    return tuple(int(x) for x in num.split("."))


def _split_sections(page_lines: list[list[tuple[str, int]]]) -> list[dict]:
    """[{label, num, kind, lines:[(line,page)]}] — 헤더 번호는 단조 증가만 허용해
    목록 항목("4. We apply ...")·표 행("32 GB 1 TB ...") 오탐을 걸러낸다."""
    sections = [{"label": "(front matter)", "num": "", "kind": "front_matter", "lines": []}]
    last_key: tuple = ()
    last_appendix = ""      # 부록 문자도 A -> B -> C 순서만 인정
    in_appendix = False
    top_kind = "other"  # 하위 절은 상위 절의 종류를 물려받는다 ("3.1 Preliminary Study" -> method)

    def start(label: str, num: str, kind: str):
        sections.append({"label": label, "num": num, "kind": kind, "lines": []})

    flat = [item for lines in page_lines for item in lines]
    for i, (line, page) in enumerate(flat):
        s = line.strip()
        next_line = flat[i + 1][0].strip() if i + 1 < len(flat) else ""
        m_special = _SPECIAL_HEADER.match(s)
        if m_special and len(s) < 40:
            label = "Impact Statement" if s.startswith("Impact") else m_special.group(1)
            start(label, "", _section_kind(label))
            in_appendix = in_appendix or label in ("References", "Bibliography")
            continue
        if (not in_appendix and len(s) < 80 and not s.endswith((".", ",", ";"))
                and (m := _NUMBERED_HEADER.match(s)) and not _UNIT_TITLE.match(m.group(2))):
            key = _num_key(m.group(1))
            if key > last_key and (not last_key or key[0] <= last_key[0] + 1):
                label = f"{m.group(1)} {m.group(2).strip()}"
                kind = _section_kind(label)
                if len(key) == 1:
                    top_kind = kind
                elif kind not in ("related_work", "conclusion") and top_kind != "other":
                    # "3.1 Preliminary Study"는 방법론 절의 일부이므로 상위 절 종류를 따른다
                    kind = top_kind
                start(label, m.group(1), kind)
                last_key = key
                continue
        if in_appendix and (m := _APPENDIX_HEADER.match(s)) and not _BIB_LINE.search(next_line):
            letter = m.group(1)[0]
            # 첫 부록은 A여야 하고, 그 뒤로는 앞으로만 진행한다 (참고문헌 속 "A Survey of ..." 오탐 방지)
            if (letter == "A" and not last_appendix) or (last_appendix and letter >= last_appendix):
                label = f"Appendix {m.group(1)} {m.group(2).strip()}"
                start(label, m.group(1), "appendix")
                last_appendix = letter
                continue
        sections[-1]["lines"].append((line, page))

    # Abstract 헤더가 없는 포맷(예: "Abstract—" 인라인)이면 앞부분을 Abstract로 취급해 잃지 않는다.
    if not any(sec["kind"] == "abstract" for sec in sections) and sections[0]["lines"]:
        sections[0].update(label="Abstract", kind="abstract")
    return [sec for sec in sections if sec["lines"]]


# --- 3) 표 / 알고리즘 블록 보호 ----------------------------------------------------
_CROSS_REF = re.compile(r"\b(Table|Figure|Section|Algorithm|Equation)\s*\d", re.I)
_FUNCTION_WORD = re.compile(r"\b(the|of|and|to|in|is|are|we|for|with|that|this|on|as|by)\b", re.I)


def _numeric_ratio(s: str) -> float:
    toks = s.split()
    return sum(1 for t in toks if re.search(r"\d", t)) / len(toks) if toks else 0.0


def _looks_like_row(s: str) -> bool:
    toks = s.split()
    if len(toks) >= 2 and _numeric_ratio(s) >= 0.5 and not _CROSS_REF.search(s):
        return True                                       # "16bit 63.88 30.76 13.50"
    # 짧은 행 라벨("Llama-2-7B", "Attention sparsity 84.3%").
    # 소문자로 시작하는 캡션 조각, "Table10and Table9" 같은 상호참조는 제외.
    return (len(toks) <= 3 and len(s) <= 40 and not s.endswith((".", ",", ";"))
            and not _CROSS_REF.search(s)
            and (s[0].isupper() or bool(re.search(r"\d", s))))


def _looks_like_prose(s: str) -> bool:
    """문장으로 보이는 줄. 열 이름만 나열한 표 헤더("Model Qasper QMSum ...")는 기능어가 없어 제외된다."""
    return (len(s.split()) >= 8 and _numeric_ratio(s) < 0.3
            and (bool(_FUNCTION_WORD.search(s)) or s.endswith((".", ":", ";"))))


def _table_end(lines: list[tuple[str, int]], start: int) -> int:
    """캡션 + 숫자 행으로 이어지는 표의 끝 인덱스(exclusive). 표가 아니면 start를 돌려준다."""
    j, caption_lines, last_row, ambiguous = start + 1, 0, -1, 0
    while j < len(lines):
        s = lines[j][0].strip()
        if _TABLE_START.match(s) or _ALGO_START.match(s) or _FIGURE_START.match(s):
            break
        if _looks_like_row(s):
            last_row, ambiguous = j, 0
        elif last_row < 0:
            caption_lines += 1
            if caption_lines > 12:          # 숫자 행이 안 나오면 표 본문이 없는 것
                return start
        elif _looks_like_prose(s):
            break
        else:
            ambiguous += 1                  # 수식 행처럼 애매한 줄은 2줄까지 표에 포함
            if ambiguous > 2:
                break
        j += 1
    return last_row + 1 if last_row >= 0 else start


def _algo_end(lines: list[tuple[str, int]], start: int) -> int:
    j = start + 1
    while j < len(lines) and j - start < 150:
        s = lines[j][0].strip()
        if _TABLE_START.match(s) or _FIGURE_START.match(s):
            break
        if not (len(s) < 70 or _ALGO_KEYWORD.match(s)) and _looks_like_prose(s):
            break
        j += 1
    return j


def _segment_blocks(lines: list[tuple[str, int]]) -> list[tuple[str, list[tuple[str, int]]]]:
    """섹션 라인들을 [(block_type, lines)] 로 나눈다. block_type: prose | table | algorithm"""
    parts, prose, i = [], [], 0

    def flush():
        nonlocal prose
        if prose:
            parts.append(("prose", prose))
            prose = []

    while i < len(lines):
        s = lines[i][0].strip()
        if _TABLE_START.match(s) and (end := _table_end(lines, i)) > i:
            flush()
            parts.append(("table", lines[i:end]))
            i = end
            continue
        if _ALGO_START.match(s) and len(s) < 80:
            end = _algo_end(lines, i)
            flush()
            parts.append(("algorithm", lines[i:end]))
            i = end
            continue
        prose.append(lines[i])
        i += 1
    flush()
    return parts


# --- 4) 청크 생성 -------------------------------------------------------------
def _split_with_pages(lines: list[tuple[str, int]], splitter) -> list[tuple[str, int]]:
    """라인 묶음을 splitter로 나누고, 각 조각의 시작 위치로 페이지를 결정한다."""
    text = "\n".join(l for l, _ in lines)
    offsets, pages, pos = [], [], 0
    for l, p in lines:
        offsets.append(pos)
        pages.append(p)
        pos += len(l) + 1
    out, cursor = [], 0
    for piece in splitter.split_text(text):
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = text.find(piece)
        if idx < 0:
            idx = cursor
        out.append((piece, pages[max(0, bisect.bisect_right(offsets, idx) - 1)]))
        cursor = max(cursor, idx + 1)   # overlap 때문에 다음 조각은 이번 조각 끝보다 앞에서 시작한다
    return out


def _short_label(label: str, limit: int = 40) -> str:
    if len(label) <= limit:
        return label
    cut = label[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _chunk_section(sec: dict, splitter, block_splitter) -> tuple[list[dict], str]:
    """한 섹션을 [{text, page, block_type, caption}] 로 만든다. 표 귀속용 산문 전체 텍스트도 함께 돌려준다."""
    chunks, prose_texts = [], []
    for block_type, lines in _segment_blocks(sec["lines"]):
        if block_type == "prose":
            prose_texts.append("\n".join(l for l, _ in lines))
            pieces = _split_with_pages(lines, splitter)
            caption = ""
        else:
            text = "\n".join(l for l, _ in lines)
            pieces = ([(text, lines[0][1])] if len(text) <= MAX_BLOCK_CHARS
                      else _split_with_pages(lines, block_splitter))
            caption = lines[0][0].strip()[:120]
        for text, page in pieces:
            # 표 사이에 낀 짧은 산문은 바로 앞 산문 청크에 붙인다
            if (block_type == "prose" and len(text) < MIN_PROSE_CHARS and chunks
                    and chunks[-1]["block_type"] == "prose"
                    and len(chunks[-1]["text"]) + len(text) <= CHUNK_SIZE + MIN_PROSE_CHARS):
                chunks[-1]["text"] += "\n" + text
                continue
            chunks.append({"text": text, "page": page, "block_type": block_type, "caption": caption})
    return chunks, "\n".join(prose_texts)


def _table_home(caption: str, prose_by_section: list[tuple[dict, str]]) -> Optional[dict]:
    """표는 PDF 레이아웃상 참조된 절과 다른 페이지에 떠다닐 수 있다 (KIVI Table 4/5는 Related Work
    페이지에 위치). 본문에서 그 표를 처음 언급한 절을 표의 소속 절로 본다."""
    m = re.match(r"Table\s*(\d+)", caption)
    if not m:
        return None
    ref = re.compile(rf"Table\s*{m.group(1)}(?!\d)")
    for sec, prose in prose_by_section:
        if ref.search(prose):
            return sec
    return None


def _build_chunks(pages: list[Document], candidate_id: str, filename: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    block_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_BLOCK_CHARS, chunk_overlap=0)

    sections = _split_sections(_clean_pages(pages))
    if len(sections) <= 1:
        # 헤더를 하나도 못 찾은 PDF -> 예전 방식(문자 단위 분할)으로 폴백
        chunks = splitter.split_documents(pages)
        for i, c in enumerate(chunks):
            c.metadata.update(source_id=candidate_id, source_file=filename, section="(unknown)",
                              section_num="", section_kind="other", block_type="prose",
                              chunk_id=f"{candidate_id}-{i:03d}")
        return chunks

    kept = [sec for sec in sections if sec["kind"] not in SKIP_SECTION_KINDS]
    chunked = [(sec, *_chunk_section(sec, splitter, block_splitter)) for sec in kept]
    prose_by_section = [(sec, prose) for sec, _, prose in chunked]

    docs: list[Document] = []
    for sec, chunks, _ in chunked:
        for c in chunks:
            home = sec
            position = ""
            if c["block_type"] == "table":
                found = _table_home(c["caption"], prose_by_section)
                if found is not None and found is not sec:
                    home, position = found, sec["label"]
            idx = len(docs)
            docs.append(Document(
                page_content=f"[§{_short_label(home['label'])}]\n{c['text']}",
                metadata={
                    "source_id": candidate_id,
                    "source_file": filename,
                    "page": c["page"],
                    "section": home["label"],
                    "section_num": home["num"],
                    "section_kind": home["kind"],
                    "block_type": c["block_type"],
                    "caption": c["caption"],
                    # 표가 참조된 절과 다른 절의 페이지에 떠 있을 때, 실제 위치한 절
                    "position_section": position,
                    "chunk_id": f"{candidate_id}-{idx:03d}",
                },
            ))
    return docs


def build_vectorstore(candidate_id: str, filename: str) -> tuple[Optional[FAISS], Optional[str]]:
    """단일 후보 논문에 대한 벡터스토어를 만들거나, 이미 있으면 재사용한다."""
    index_path = os.path.join(PERSIST_DIR, f"cand_{candidate_id}_{INDEX_VERSION}")

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        # 인덱스와 함께 저장되는 docstore가 pickle이라 명시적 허용이 필요하다.
        # 이 파일은 항상 로컬에서 직접 생성한 것이므로 외부 입력이 아니다.
        vs = FAISS.load_local(
            index_path, get_embeddings(), allow_dangerous_deserialization=True
        )
        return vs, None

    pages, warning = _load_pdf(filename)
    if pages is None:
        # PDF도 없고 기존 벡터스토어도 없음 -> 임베딩 모델 로드 자체를 건너뛴다.
        return None, warning

    embeddings = get_embeddings()

    if len(pages) > MAX_TOTAL_PAGES:
        warning = (
            f"[RAG] {filename} 이 {len(pages)}페이지로 과제 한도(200p)를 초과합니다. "
            f"앞 {MAX_TOTAL_PAGES}페이지만 사용합니다."
        )
        pages = pages[:MAX_TOTAL_PAGES]

    chunks = _build_chunks(pages, candidate_id, filename)

    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(index_path)
    return vs, warning

