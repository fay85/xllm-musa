import unittest

from long_verify_graph_budget import summarize


def event(name, cat, ts, dur, corr=None, tid=4):
    return dict(name=name, cat=cat, ts=ts, dur=dur, ph="X", pid=3, tid=tid,
                args={} if corr is None else {"correlation": corr})


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.events = [event("V41/step/q=6/kv=2010", "user_annotation", 0, 5),
                       event("musaGraphLaunch", "runtime", 1, 1, 17),
                       event("musaGraphLaunch", "runtime", 2, 1, 18, tid=7),
                       event("draft", "kernel", 20, 1000, 18),
                       event("indexer", "kernel", 30, 10, 17),
                       event("other", "kernel", 35, 10, 17)]

    def test_async_union_and_identity(self):
        s = summarize(self.events)
        self.assertEqual(s["mean_union_ms"], 0.015)
        self.assertEqual(s["graphs"][0]["nodes"], 2)
        self.assertEqual({k["name"] for k in s["kernels"]}, {"indexer", "other"})

    def test_missing_nodes(self):
        with self.assertRaises(ValueError):
            summarize(self.events[:3])

    def test_ambiguous(self):
        with self.assertRaises(ValueError):
            summarize(self.events + [self.events[1]])
