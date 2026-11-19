import json
import unittest

from long_router_ingress import budgets


class RouterTest(unittest.TestCase):
    def setUp(self):
        self.audit = {"warmup": [{"id": "reply"}], "formal": []}
        self.terminal = ["record_type: sequence_terminal, request_id: reply, x-request-id: uuid,"]
        def line(ms, kind, result, ns=None):
            v = dict(global_request_id="uuid", owner_role="EVENT_OWNER_ROLE_SERVICE",
                     event_type="REQUEST_EVENT_TYPE_" + kind, result="EVENT_RESULT_" + result,
                     stage_duration_validity="METRIC_VALIDITY_" +
                     ("VALID" if ns is not None else "NOT_APPLICABLE"), stage_duration_ns=ns)
            return f"2026-11-19 19:55:30.{ms:03d} event=request_lifecycle payload=" + json.dumps(v)
        self.router = [line(0, "INGRESS", "STARTED"),
                       line(191, "PREFILL_QUEUE", "ACCEPTED", 191000001),
                       line(202, "ROUTE", "SUCCEEDED", 11200000),
                       line(202, "ROUTE", "SUCCEEDED"),
                       line(203, "P_DISPATCH", "ACCEPTED")]

    def test_valid_with_non_duration_route_record(self):
        row, = budgets(self.router, self.terminal, self.audit)
        self.assertEqual((row["ingress_to_queue_ms"], row["queue_to_route_ms"],
                          row["ingress_to_dispatch_log_ms"]), (191.000001, 11.2, 203))

    def test_ambiguous_or_missing_join(self):
        for router, terms in [(self.router + [self.router[2]], self.terminal),
                              (self.router, []), (self.router, self.terminal * 2)]:
            with self.subTest(), self.assertRaises(ValueError):
                budgets(router, terms, self.audit)
