"""Join EvalScope streaming rows to native C1 sequence_terminals by completion id.

No network, no changes to measurements. Exit nonzero for incomplete, duplicate,
or mismatched groups (including warmup). Do not use request_id from the DB:
it can be NULL even when SSE chunks have a server-issued id.
"""
import argparse
import base64
import hashlib
import io
import json
import pickle
import re
import sqlite3
from pathlib import Path


class PlainData(pickle.Unpickler):
    """EvalScope stores JSON-like SSE events in base64-pickled lists."""

    def find_class(self, module, name):
        raise ValueError("non-data pickle in benchmark DB")

    def persistent_load(self, pid):
        raise ValueError("unexpected persistent object")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def content_only(events, rid, *, native_closing_markers=0):
    """Inspect preserved SSE JSON; usage can be a separate event or absent from EvalScope DB."""
    pieces = []
    finish = None
    for event in events:
        choices = event.get("choices", [])
        require(len(choices) <= 1, f"multiple choices in C1 sample {rid}")
        for choice in choices:
            require(event.get("id") == rid and choice.get("index", 0) == 0, f"unbound choice {rid}")
            require(finish is None, f"choice after finish {rid}")
            delta = choice.get("delta", {})
            require(not any(delta.get(key) for key in ("reasoning_content", "reasoning", "reasoning_details",
                                                       "tool_calls", "function_call")), f"non-content output {rid}")
            pieces.append(delta.get("content") or "")
            finish = choice.get("finish_reason") or None
        usage = event.get("usage") or {}
        for source in (usage, usage.get("completion_tokens_details") or {}):
            if "reasoning_tokens" in source and source["reasoning_tokens"] is not None:
                n = source["reasoning_tokens"]
                require(type(n) is int and n == 0, f"nonzero or invalid reasoning usage {rid}")
    text = "".join(pieces)
    require(finish == "length" and text, f"incomplete content stream {rid}")
    closing = list(re.finditer(r"</think>", text, re.IGNORECASE))
    require(not re.search(r"<think>", text, re.IGNORECASE)
            and len(closing) <= native_closing_markers, f"reasoning delimiter in body {rid}")
    if closing:
        require(text[:closing[0].start()].strip(), f"closing delimiter is the first body output {rid}")
    return text


def read_wire(path: Path, size: int) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    by_time = {record['start_time']: record for record in records}
    require(len(by_time) == len(records) == size, f'{path}: wrong wire record count')
    return by_time


def check_wire(record: dict, fixture: dict, persisted: list[dict], rid: str,
               length: int, outputs: int) -> list[dict]:
    body = json.loads(record['request'])
    require('token_ids' not in body and all(body.get(k) == v for k, v in fixture.items()),
            f'{rid}: posted request differs from the textual fixture')
    raw = base64.b64decode(record['base64_sse'], validate=True).decode()
    events = []
    done = False
    for line in raw.splitlines():
        if not line or line.startswith(':'):
            continue
        require(line.startswith('data:') and not done, f'{rid}: unexpected SSE data')
        payload = line[5:].strip()
        if payload == '[DONE]':
            done = True
        else:
            event = json.loads(payload)
            require(not event.get('error') and event.get('object') != 'error', f'{rid}: SSE error')
            events.append(event)
    require(done and {event['id'] for event in events if event.get('id')} == {rid},
            f'{rid}: no SSE completion or inconsistent IDs')
    require([event for event in events if event.get('choices')] == persisted,
            f'{rid}: raw SSE choices differ from EvalScope')
    first = next((choice['delta']['content'] for event in events for choice in event.get('choices', [])
                  if choice.get('delta', {}).get('content')), None)
    require(first and first.strip() and not first.lstrip().lower().startswith(('</think>', '<think>')),
            f'{rid}: first nonempty output is not body text')
    usages = [event['usage'] for event in events if event.get('usage')]
    require(len(usages) == 1 and (usages[0]['prompt_tokens'], usages[0]['completion_tokens']) == (length, outputs),
            f'{rid}: SSE usage missing or inconsistent')
    for stats in (usages[0], usages[0].get('prompt_tokens_details') or {}):
        if stats.get('cached_tokens') is not None:
            require(type(stats['cached_tokens']) is int and stats['cached_tokens'] == 0,
                    f'{rid}: SSE prefix hit disagrees with native terminal')
    return events


def native_proof(fields: dict[str, str], rid: str, prompt_ids_digest: str,
                 required: bool) -> dict | None:
    keys = ('ttft20_prompt_ids_sha256', 'ttft20_prompt_last_thinking_marker',
            'ttft20_output_open_thinking_markers', 'ttft20_output_close_thinking_markers',
            'ttft20_reasoning_tokens')
    if not any(key in fields for key in keys):
        require(not required, f'{rid}: missing native tokenization and reasoning proof')
        return None
    require(all(key in fields for key in keys), f'{rid}: incomplete native proof')
    actual_digest, marker, opening, closing, reasoning = (fields[key] for key in keys)
    require(actual_digest == prompt_ids_digest and int(marker) == 128822,
            f'{rid}: native prompt IDs or final control marker changed')
    require(int(opening) == 0 and int(reasoning) == 0 and 0 <= int(closing) <= 1024,
            f'{rid}: unexpected reasoning tokens or control markers')
    return dict(actual_prompt_ids_sha256=actual_digest, prompt_last_thinking_marker=int(marker),
                output_thinking_open_markers=int(opening), output_thinking_close_markers=int(closing),
                reasoning_tokens=int(reasoning))


def audit(directory: Path, log: Path, length: int, outputs: int, count: int,
          *, wire: bool = False, require_native: bool = False) -> dict[str, list]:
    require(not require_native or wire, 'native qualification requires raw SSE')
    identity = json.loads((directory/'identity.json').read_text()) if wire else {}
    terminals = {}
    requests = {}
    for line in log.open(errors="replace"):
        is_request = wire and "record_type: request_terminal," in line
        if not is_request and "record_type: sequence_terminal," not in line:
            continue
        fields = dict(re.findall(r"(?:^|, )([\w-]+): ([^,\n]+)", line))
        rid = fields["request_id"]
        index = requests if is_request else terminals
        require(rid not in index, f"duplicate server terminal {rid}")
        index[rid] = fields

    seen = set()
    report = {}
    for phase, size in (("warmup", 1), ("formal", count)):
        dbs = list((directory / phase).rglob("benchmark_data.db"))
        require(len(dbs) == 1, f"{phase}: missing or ambiguous DB")
        with sqlite3.connect(dbs[0].resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = list(db.execute("SELECT * FROM result ORDER BY start_time"))
        require(len(rows) == size, f"{phase}: {len(rows)} != {size}")
        wire_records = read_wire(directory/f'wire_{phase}.jsonl', size) if wire else None
        report[phase] = []
        for row in rows:
            require(row["success"] == 1 and row["prompt_tokens"] == length and
                    row["completion_tokens"] == outputs, "failed request or wrong length")
            events = PlainData(io.BytesIO(base64.b64decode(row["response_messages"],
                                                           validate=True))).load()
            ids = {e["id"] for e in events if e.get("id")}
            require(len(ids) == 1, f"inconsistent completion identity {ids}")
            rid = ids.pop()
            require(rid in terminals and rid not in seen, f"unmatched or repeated {rid}")
            seen.add(rid)
            fields = terminals[rid]
            if wire:
                request = requests.pop(rid)
                require(request['outcome'] == 'success' and request['status_code'] == '0' and
                        request['x-request-id'] == fields['x-request-id'] and fields['sequence'] == '0',
                        f'{rid}: request and sequence terminal do not agree')
                events = check_wire(wire_records.pop(row['start_time'].hex()), identity['request_body'],
                                    events, rid, length, outputs)
            proof = native_proof(fields, rid, identity['token_ids_sha256'], require_native) if wire else None
            text = content_only(events, rid, native_closing_markers=proof['output_thinking_close_markers'] if proof else 0)
            usages = [e["usage"] for e in events if e.get("usage")]
            # EvalScope can omit the usage-only SSE from response_messages.
            # The result row still carries usage; cross-check the native terminal.
            if usages:
                require(usages[-1]["prompt_tokens"] == length and
                        usages[-1]["completion_tokens"] == outputs, f"usage mismatch {rid}")
            require(fields["finish_reason"] == "length" and
                    int(fields["num_prefix_cache_tokens"]) == 0 and
                    int(fields["prompt_tokens"]) == length and
                    int(fields["generated_tokens"]) == outputs, f"terminal mismatch {rid}")
            require(0 <= int(fields["speculative_accepted_tokens"]) <=
                    int(fields["speculative_proposed_tokens"]), f"invalid draft counts {rid}")
            ttft, latency = row["first_chunk_latency"], row["latency"]
            require(0 < ttft < latency, f"bad timing {rid}")
            report[phase].append(dict(id=rid, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      accepted=int(fields["speculative_accepted_tokens"]),
                                      proposed=int(fields["speculative_proposed_tokens"]),
                                      ttft_s=ttft, e2e_s=latency,
                                      tpot_ms=1000*(latency-ttft)/(outputs-1)))
            if wire:
                report[phase][-1].update(raw_sse=True, native_token_proof=proof)
    require(seen == set(terminals), "server log contains extra requests; supply isolated log")
    require(not requests, "server log contains extra request terminals")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("decode_rank0_log", type=Path)
    ap.add_argument("--length", required=True, type=int)
    ap.add_argument("--outputs", default=1024, type=int)
    ap.add_argument("--count", required=True, type=int)
    ap.add_argument("--wire", action="store_true")
    ap.add_argument("--require-native", action="store_true")
    args = ap.parse_args()
    print(json.dumps(audit(args.directory, args.decode_rank0_log, args.length,
                           args.outputs, args.count, wire=args.wire, require_native=args.require_native), indent=2))
