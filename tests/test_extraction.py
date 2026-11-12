import copy
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bpc_gnn.bpc_adapter import BPC_ROOT, FisherScoreReference, continuation_errors, score_candidates
from bpc_gnn.dataset import DemonstrationBank
from bpc_gnn.evaluate import evaluate
from bpc_gnn.model import GraphRetriever, ModelConfig, encode_bank, replay_bank_gradients
from bpc_gnn.prepare import annotation_windows, demo_split
from bpc_gnn.selection import NonemptyBernoulli
from bpc_gnn.train import train

sys.path.insert(0, str(BPC_ROOT))
from dpc_nldl.summary_data import annotation_windows as original_windows
from dpc_nldl.summary_retrieval import NonemptyBernoulli as OriginalBernoulli
from dpc_nldl.summary_fisher_residual import evaluate as original_evaluate
from tests import test_summary_fisher_residual as reference_tests
from tests.support.policies import trajectories, fisher_recipe


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_model_scores_gradients_and_evaluation_match_original(self):
        bank, query, reference, original = reference_tests.FisherResidualTest().fixture()
        model = GraphRetriever(ModelConfig(**{name: getattr(original.config, name)
                                              for name in ModelConfig.__dataclass_fields__}))
        with torch.no_grad():
            original.score.weight.normal_(0, .01)
        model.load_state_dict(original.state_dict())
        keys = encode_bank(model, bank, 31)
        old_keys = keys.detach().clone().requires_grad_(True)
        eligible, actual = score_candidates(model, reference, query, keys, checkpoint_graph=True)
        with original.query_context(query):
            expected = original(query.history, query.instruction, old_keys[eligible], checkpoint_graph=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(keys.grad, old_keys.grad, rtol=1e-6, atol=1e-8)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(original.named_parameters())[name].grad,
                                       rtol=1e-6, atol=1e-8, msg=name)
        config = dict(mixed_precision=False, temperature=.03, evaluation_samples=16)
        result = evaluate(model, bank, reference, keys.detach(), [query], config, 29)
        from contextlib import nullcontext
        previous = original_evaluate(original, bank, old_keys.detach(), [query], nullcontext,
                                     samples=16, temperature=.03, seed=29)
        self.assertAlmostEqual(result['mean']['action_mse'], previous['mse'], places=12)
        self.assertAlmostEqual(result['mean']['sampled_action_mse'], previous['sampled_mse'], places=12)
        changed = keys.detach().clone()
        changed[bank.policy.hankel_trajectory == query.exclude_demo] += 1e6
        self.assertEqual(result, evaluate(model, bank, reference, changed, [query], config, 29))

    def test_gradient_replay_matches_joint_encoder_autograd(self):
        bank, query, reference, original = reference_tests.FisherResidualTest().fixture()
        model = GraphRetriever(ModelConfig(**{name: getattr(original.config, name)
                                              for name in ModelConfig.__dataclass_fields__}))
        with torch.no_grad():
            model.score.weight.normal_(0, .01)
        direct = copy.deepcopy(model)
        keys = encode_bank(model, bank, 31)
        score_candidates(model, reference, query, keys)[1].sum().backward()
        replay_bank_gradients(model, bank, keys.grad, 31)
        direct_keys = direct.candidate(*bank.inputs(bank.columns))
        score_candidates(direct, reference, query, direct_keys)[1].sum().backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad,
                                       rtol=1e-5, atol=1e-6, msg=name)

    def test_sampler_and_annotation_alignment(self):
        logits = torch.tensor([1000., -100., .3, -.9], dtype=torch.float64)
        new = NonemptyBernoulli(logits)
        old = OriginalBernoulli(logits)
        samples = new.sample(100, torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(samples, old.sample(100, torch.Generator().manual_seed(1))))
        self.assertFalse(bool(samples[:, 1].any()))
        torch.testing.assert_close(new.log_prob(samples), old.log_prob(samples), rtol=0, atol=0)
        annotation = dict(steps=12, segments=[
            dict(start_step=0, end_step_exclusive=5, primitive='reach', caption='Reach the object'),
            dict(start_step=5, end_step_exclusive=12, primitive='lift', caption='Lift the object')])
        texts, indices = annotation_windows(annotation, 12, 2, 2)
        expected_texts, expected_indices = original_windows(annotation, 12, 2, 2)
        self.assertEqual(texts, expected_texts)
        np.testing.assert_array_equal(indices, expected_indices)
        self.assertEqual(set(demo_split(4, 1)), {'train', 'validation', 'test'})

    def test_training_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = []
            noise = np.random.default_rng(7)
            for task in ['a', 'b']:
                data = []
                for number, row in enumerate(trajectories(demos=4, length=12)):
                    row = {name: value+noise.normal(0, .2, value.shape) for name, value in row.items()}
                    data.append(dict(Xro=row['x'], Uro=row['u'], r3m_agent=row['agent'],
                                     r3m_wrist=row['wrist'], demo_id=str(number)))
                    records.append(dict(task=task, bank_index=number, demo=f'demo_{number}', length=12,
                        split=['train', 'train', 'validation', 'test'][number], goal_text_id=0,
                        continuation_text_ids=torch.full((8,), number+1, dtype=torch.long)))
                torch.save(dict(data=data), root/f'{task}.pt')
            torch.manual_seed(1)
            index = dict(version=1, state_offset=1, history=2, horizon=2, seed=1,
                text_embeddings=torch.randn(5, 4), demonstrations=records, text_encoder={},
                instructions={'a': 'a', 'b': 'b'}, annotation_identity='fixture')
            torch.save(index, root/'index.pt')
            base = dict(method=fisher_recipe(), data=dict(state_key='Xro', action_key='Uro',
                        feature_keys=['r3m_agent', 'r3m_wrist'], r3m=dict(feature_dim=1)))
            base['method']['g']['lda']['anchors'] = 8
            bank = DemonstrationBank(root, root/'index.pt', base, device='cpu')
            from scripts.evaluate_summary_fisher_reference import install_fisher as initialize_fisher
            initialize_fisher(bank.policy)
            bank.policy.fit_g_lookup()
            reference = FisherScoreReference.from_installed_bank(bank)
            torch.save(dict(data_identity=bank.identity, lda=bank.policy.support_selector.payload,
                matrix=bank.policy.g_lookup_matrix, stats=bank.policy.g_lookup_stats), root/'fisher.pt')
            reference.fisher_sha256 = hashlib.sha256((root/'fisher.pt').read_bytes()).hexdigest()
            config = json.loads((Path(__file__).parents[1]/'configs/train.json').read_text())
            config.update(steps=2, queries_per_update=2, samples_per_query=8, evaluation_samples=4, temperature=.1,
                          mixed_precision=False, bank_chunk_size=7, checkpoint_every=1,
                          validation_every=1, validation_per_task=1, probe_queries_per_task=1)
            config['model'] = dict(history_dim=8, text_dim=4, hidden_dim=12, width=8, heads=2, summaries=3)
            (root/'base.json').write_text(json.dumps(base))
            (root/'config.json').write_text(json.dumps(config))
            train(bank, reference, config, root/'resumed')
            config['steps'] = 3
            resumed = train(bank, reference, config, root/'resumed', resume=True)
            whole = train(bank, reference, config, root/'whole')
            self.assertGreater(float(resumed.score.weight.detach().norm()), 1e-8)
            for name, parameter in resumed.state_dict().items():
                torch.testing.assert_close(parameter, whole.state_dict()[name], rtol=0, atol=0)
            records = [json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in records if r['event']=='train'], [1, 2, 3])
            self.assertTrue(all([q['task'] for q in r['queries']]==['a', 'b']
                                for r in records if r['event']=='train'))
            with self.assertRaises(ValueError):
                train(bank, reference, config, root/'resumed')
            from bpc_gnn.evaluate import main as evaluate_main
            from bpc_gnn.train import main as train_main
            arguments = ['--banks', str(root), '--annotations', str(root/'index.pt'),
                '--fisher', str(root/'fisher.pt'), '--fisher-cache', str(root/'keys.pt'),
                '--base-config', str(root/'base.json'), '--device', 'cpu']
            with patch.object(sys, 'argv', ['evaluate', *arguments,
                    '--checkpoint', str(root/'resumed/checkpoint.pt'), '--output', str(root/'evaluation.json')]):
                evaluate_main()
            self.assertEqual(len(json.loads((root/'evaluation.json').read_text())['rows']), 2)
            with patch.object(sys, 'argv', ['train', *arguments,
                    '--config', str(root/'config.json'), '--output', str(root/'cli'), '--steps', '1']):
                train_main()
            self.assertTrue(json.loads((root/'cli/status.json').read_text())['complete'])



if __name__ == '__main__':
    unittest.main()
