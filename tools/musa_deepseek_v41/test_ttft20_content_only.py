"""Offline regression for JSON-stream validation shared by native and SG."""
import base64
import json
import unittest
from audit_long_c1 import check_wire, content_only, native_proof
def chunk(s='',reason=None,rid='x',index=0,**kwargs):
    return {'id':rid,'choices':[{'index':index,'delta':{'content':s,**kwargs},'finish_reason':reason}]}
class ContentTest(unittest.TestCase):
    def test_non_scored_usage_tail(self):
        self.assertEqual(content_only([chunk('body','length'),{'usage':{'completion_tokens_details':{'reasoning_tokens':0}}}],'x'),'body')
    def test_split_tags(self):
        for parts in [('<thi','nk>'),('</Th','Ink>')]:
            with self.assertRaises(ValueError):content_only([chunk(parts[0]),chunk(parts[1],'length')],'x')
    def test_native_redundant_closing_markers(self):
        rows=[chunk('body</th'),chunk('ink> continues','length')]
        self.assertEqual(content_only(rows,'x',native_closing_markers=1),'body</think> continues')
        with self.assertRaises(ValueError):content_only(rows,'x')
        initial=[chunk('</think>'),chunk('body','length')]
        with self.assertRaises(ValueError):content_only(initial,'x',native_closing_markers=1)
        opened=[chunk('body<think>'),chunk('more','length')]
        with self.assertRaises(ValueError):content_only(opened,'x',native_closing_markers=1)
    def test_inconsistent_source_or_finish(self):
        for rows in [[chunk('body','length'),chunk('late')],[chunk('why','length',rid=None)],
                     [chunk('body','length',index=1)],[chunk('yes','length',reasoning_content='reason')],
                     [chunk('yes','length'),{'usage':{'completion_tokens_details':{'reasoning_tokens':1}}}]]:
            with self.assertRaises(ValueError):content_only(rows,'x')

    def test_wire_requires_done_usage_and_text_body(self) -> None:
        fixture = {'messages': [{'role': 'user', 'content': 'some text'}]}
        choices = [chunk('body', 'length')]
        usage = {'id': 'x', 'choices': [], 'usage': {'prompt_tokens': 20000, 'completion_tokens': 1024}}

        def evidence(items: list[dict], done: bool = True, **body: object) -> dict:
            raw = ''.join(f'data: {json.dumps(item)}\n\n' for item in items)
            if done:
                raw += 'data: [DONE]\n\n'
            return {'request': json.dumps(dict(fixture, **body)), 'base64_sse': base64.b64encode(raw.encode())}

        events = choices + [usage]
        self.assertEqual(check_wire(evidence(events), fixture, choices, 'x', 20000, 1024), events)
        for invalid in (evidence(choices), evidence(events, done=False), evidence(events, token_ids=[8])):
            with self.assertRaises(ValueError):
                check_wire(invalid, fixture, choices, 'x', 20000, 1024)

    def test_missing_native_data_never_counts_as_zero(self) -> None:
        self.assertIsNone(native_proof({}, 'x', 'sealed', False))
        with self.assertRaises(ValueError):
            native_proof({}, 'x', 'sealed', True)
        proof = dict(ttft20_prompt_ids_sha256='sealed', ttft20_prompt_last_thinking_marker='128822',
                     ttft20_output_open_thinking_markers='0', ttft20_output_close_thinking_markers='0',
                     ttft20_reasoning_tokens='0')
        self.assertEqual(native_proof(proof, 'x', 'sealed', True)['reasoning_tokens'], 0)
        self.assertRaises(ValueError, native_proof, proof, 'x', 'other', True)
