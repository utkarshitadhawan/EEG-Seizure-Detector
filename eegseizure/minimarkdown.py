"""A tiny Markdown reader for the two documents shown in the GUI (docs/USER_GUIDE.md and TRAINING_REPORT.md) and for HTML export.

Supported: # / ## / ### headings, paragraphs, "- " bullets, pipe tables, fenced code blocks (~~~ or three backticks),
![alt](path) images on their own line, and inline **bold** and `code`. Anything else is shown as plain text.
"""
import base64
import html
import os
import re

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def parse(text):
    """Return a list of blocks: ('h1'|'h2'|'h3'|'p'|'li', text) | ('table', rows) | ('code', text) | ('img', (alt, path))."""
    blocks, para, lines, i = [], [], text.splitlines(), 0

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith(("~~~", "```")):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith(("~~~", "```")):
                code.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code)))
        elif re.match(r"^#{1,3} ", s):
            flush()
            level = len(s.split(" ")[0])
            blocks.append((f"h{level}", s[level + 1:]))
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):          # skip the |---|---| separator row
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        elif re.match(r"^!\[[^\]]*\]\([^)]+\)$", s):
            flush()
            m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
            blocks.append(("img", (m.group(1), m.group(2))))
        elif s.startswith("- "):
            flush()
            blocks.append(("li", s[2:]))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return blocks


def split_inline(text):
    """Split a line into (kind, text) pieces; kind is 'text', 'bold' or 'code'."""
    out = []
    for part in INLINE.split(text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(("bold", part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(("code", part[1:-1]))
        elif part:
            out.append(("text", part))
    return out


def _inline_html(text):
    out = ""
    for kind, t in split_inline(text):
        t = html.escape(t)
        out += f"<b>{t}</b>" if kind == "bold" else f"<code>{t}</code>" if kind == "code" else t
    return out


def to_html(text, base_dir=".", title="Report"):
    """Render Markdown-lite to a single self-contained HTML page (images are embedded)."""
    body, in_list = [], False
    for kind, val in parse(text):
        if kind != "li" and in_list:
            body.append("</ul>")
            in_list = False
        if kind in ("h1", "h2", "h3"):
            body.append(f"<{kind}>{_inline_html(val)}</{kind}>")
        elif kind == "p":
            body.append(f"<p>{_inline_html(val)}</p>")
        elif kind == "li":
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline_html(val)}</li>")
        elif kind == "code":
            body.append(f"<pre>{html.escape(val)}</pre>")
        elif kind == "table":
            head, rest = val[0], val[1:]
            body.append("<table><tr>" + "".join(f"<th>{_inline_html(c)}</th>" for c in head) + "</tr>"
                        + "".join("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>" for r in rest) + "</table>")
        elif kind == "img":
            alt, rel = val
            p = os.path.join(base_dir, rel)
            if os.path.isfile(p):
                body.append(f'<p><img alt="{html.escape(alt)}" style="max-width:100%" src="data:image/png;base64,'
                            f'{base64.b64encode(open(p, "rb").read()).decode()}"></p>')
    if in_list:
        body.append("</ul>")
    css = ("body{font-family:Arial,Helvetica,sans-serif;max-width:980px;margin:24px auto;padding:0 12px;line-height:1.45}"
           "table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #bbb;padding:4px 8px;font-size:14px}"
           "th{background:#eee}code,pre{background:#f4f4f4;padding:2px 4px}pre{padding:8px;overflow:auto}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(body)}</body></html>"
