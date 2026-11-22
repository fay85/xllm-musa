# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest

from long_20k_200k_gate import canonical_sha, judge


class LongGoalGateTest(unittest.TestCase):
    def setUp(self) -> None:
        audits = []
        for group in (0, 1):
            rows = [dict(id=f'ref{group}-{i}', phase='warmup' if i == 0 else 'formal',
                         ttft_ms=2000.0, tpot_ms=10.0, e2e_s=12.23) for i in range(6)]
            audits.append(dict(length=200000, count=5, token_ids_sha256='input200k', samples=rows))
        self.reference = dict(length=200000, outputs=1024, token_ids_sha256='input200k',
                              dataset_sha256='dataset200k', raw_audits=audits,
                              audit_sha256=[canonical_sha(a) for a in audits])
        self.groups = []
        for group, phase in enumerate(('F', 'R')):
            rows = [dict(id=f'candidate{group}-{i}', ttft_s=1.0, e2e_s=9.184,
                         tpot_ms=8.0) for i in range(6)]
            starts = {role: dict(host='sj9' if role == 'P' else 'sj10', container=role,
                                pid=10 + group, start_ticks=str(100 + group),
                                at=f'2026-11-20T0{group + 1}:00:00+08:00')
                      for role in ('P', 'D', 'router')}
            self.groups.append(dict(uid=str(group), phase=phase, starts=starts,
                                    execution_config_sha256='same', assets={'ELF': 'same'},
                                    identity=dict(length=200000, output_tokens=1024, concurrency=1,
                                                  warmup=1, formal=5, profiler_on=False,
                                                  token_ids_sha256='input200k', dataset_sha256='dataset200k'),
                                    released={'sj9': True, 'sj10': True}, basic_answers=['4.2', '56'],
                                    raw_audit={'length': 200000, 'outputs': 1024, 'source_sha256': {'db': 'hash'}},
                                    audit={'warmup': rows[:1], 'formal': rows[1:]}))
        registration = dict(at='2026-11-20T00:00:00+08:00', length=200000, outputs=1024,
                            rounds=[dict(phase=g['phase'], uid=g['uid']) for g in self.groups])
        registration['sha256'] = canonical_sha(registration)
        for group in self.groups:
            group['registration'] = copy.deepcopy(registration)

    def test_pass_two_independent_rounds(self) -> None:
        self.assertTrue(judge(self.reference, self.groups)['pass'])

    def test_inclusive_ttft_and_tpot_strict_e2e(self) -> None:
        for mode in ('ttft_equal', 'tpot_equal', 'e2e_equal'):
            groups = copy.deepcopy(self.groups)
            for group in groups:
                for row in group['audit']['formal']:
                    row['ttft_s'] = 2.0 if mode != 'tpot_equal' else 1.0
                    row['tpot_ms'] = 8.0 if mode == 'ttft_equal' else 10.0
                    row['e2e_s'] = row['ttft_s'] + row['tpot_ms'] * 1023 / 1000
            self.assertEqual(judge(self.reference, groups)['pass'], mode != 'e2e_equal')

    def test_failures_cannot_be_averaged_away(self) -> None:
        groups = copy.deepcopy(self.groups)
        for row in groups[1]['audit']['formal']:
            row.update(ttft_s=3.0, tpot_ms=10.0, e2e_s=13.23)
        result = judge(self.reference, groups)
        self.assertTrue(result['rounds'][0]['pass'])
        self.assertFalse(result['pass'])

    def test_reject_missing_drift_duplicate_or_truncated(self) -> None:
        mutations = [
            lambda g: g[1]['starts'].__setitem__('router', g[0]['starts']['router']),
            lambda g: g[1]['starts'].__setitem__('P', g[0]['starts']['P']),
            lambda g: g[1]['starts'].__setitem__('D', g[0]['starts']['D']),
            lambda g: g[1].__setitem__('execution_config_sha256', 'changed'),
            lambda g: g[1].__setitem__('assets', {'ELF': 'changed'}),
            lambda g: g[1]['released'].__setitem__('sj9', False),
            lambda g: g[1]['audit']['warmup'].clear(),
            lambda g: g[1]['audit']['formal'].pop(),
            lambda g: g[1]['audit']['formal'][0].__setitem__('id', g[0]['audit']['formal'][0]['id']),
            lambda g: g[1]['identity'].__setitem__('length', 20000),
            lambda g: g[1]['raw_audit'].__setitem__('length', 20000),
            lambda g: g[1]['identity'].__setitem__('token_ids_sha256', 'wrong'),
            lambda g: g[1]['identity'].__setitem__('output_tokens', 1000),
            lambda g: g[1]['identity'].__setitem__('profiler_on', True),
            lambda g: g[1]['audit']['formal'][0].__setitem__('ttft_s', float('nan')),
            lambda g: g[1]['audit']['formal'][0].__setitem__('tpot_ms', 0.01),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                groups = copy.deepcopy(self.groups)
                mutation(groups)
                with self.assertRaises(ValueError):
                    judge(self.reference, groups)

    def test_wrong_reference_identity_or_sha(self) -> None:
        for length in (256, 2000, 20000):
            reference = copy.deepcopy(self.reference)
            reference['length'] = length
            with self.assertRaises(ValueError):
                judge(reference, self.groups)
        reference = copy.deepcopy(self.reference)
        reference['raw_audits'][0]['samples'][0]['ttft_ms'] += 1
        with self.assertRaises(ValueError):
            judge(reference, self.groups)

    def test_registration_must_precede_both_rounds(self) -> None:
        groups = copy.deepcopy(self.groups)
        for group in groups:
            registration = group['registration']
            registration['at'] = '2026-11-20T01:30:00+08:00'
            registration['sha256'] = canonical_sha(
                {k: v for k, v in registration.items() if k != 'sha256'})
        with self.assertRaises(ValueError):
            judge(self.reference, groups)
        groups = copy.deepcopy(self.groups)
        groups[0].pop('registration')
        with self.assertRaises(ValueError):
            judge(self.reference, groups)


if __name__ == '__main__':
    unittest.main()
