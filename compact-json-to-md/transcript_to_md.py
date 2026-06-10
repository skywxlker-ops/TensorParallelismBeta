import json
import re
import sys

def extract_blocks(file_path):
    """Parse the === ROLE === / [json array] alternating format."""
    with open(file_path, "r") as f:
        text = f.read()

    # Split on role markers
    pattern = re.compile(r"=== (USER|ASSISTANT) ===\n(.*?)(?=\n=== |\Z)", re.DOTALL)
    blocks = []
    for match in pattern.finditer(text):
        role = match.group(1)
        content_str = match.group(2).strip()
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            continue
        blocks.append((role, content))
    return blocks


def render_markdown(blocks):
    md = []
    md.append("# Conversation Transcript -- BluBridge BLAS Kernel Analysis\n")

    turn_num = 0
    pending_tool_calls = {}  # tool_use_id -> tool input

    for role, content in blocks:
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            itype = item.get("type")

            # Skip thinking blocks (encrypted, not useful)
            if itype == "thinking":
                continue

            if itype == "text":
                text = item.get("text", "").strip()
                if not text:
                    continue
                if role == "USER":
                    turn_num += 1
                    md.append(f"\n---\n\n## Turn {turn_num} -- User\n")
                    md.append(text)
                    md.append("")
                else:  # ASSISTANT
                    md.append(f"\n### Assistant\n")
                    md.append(text)
                    md.append("")

            elif itype == "tool_use":
                tool_id = item.get("id", "")
                tool_name = item.get("name", "tool")
                tool_input = item.get("input", {})
                pending_tool_calls[tool_id] = tool_input

                cmd = tool_input.get("command", "")
                desc = tool_input.get("description", "")
                file_path = tool_input.get("file_path", "")
                file_content = tool_input.get("content", "")

                md.append(f"\n#### Tool Call -- `{tool_name}`")
                if desc:
                    md.append(f"_{desc}_")
                if cmd:
                    md.append("```bash")
                    md.append(cmd)
                    md.append("```")
                if file_path:
                    md.append(f"**File:** `{file_path}`")
                if file_content:
                    md.append("```")
                    md.append(file_content[:5000])  # cap big writes
                    if len(file_content) > 5000:
                        md.append(f"\n... [truncated -- full length: {len(file_content)} chars]")
                    md.append("```")
                md.append("")

            elif itype == "tool_result":
                result_content = item.get("content", "")
                is_error = item.get("is_error", False)

                # Result content can be str or list of blocks
                if isinstance(result_content, list):
                    parts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and rc.get("type") == "text":
                            parts.append(rc.get("text", ""))
                        elif isinstance(rc, str):
                            parts.append(rc)
                    result_content = "\n".join(parts)

                tag = "Tool Result (ERROR)" if is_error else "Tool Result"
                md.append(f"\n#### {tag}")
                md.append("```")
                # Cap very long outputs
                if len(result_content) > 8000:
                    md.append(result_content[:4000])
                    md.append(f"\n... [truncated -- full length: {len(result_content)} chars] ...\n")
                    md.append(result_content[-4000:])
                else:
                    md.append(result_content)
                md.append("```")
                md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Contents.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "transcript.md"

    blocks = extract_blocks(src)
    md = render_markdown(blocks)

    with open(dst, "w") as f:
        f.write(md)

    print(f"Done -- {len(blocks)} blocks parsed --> {dst}")
