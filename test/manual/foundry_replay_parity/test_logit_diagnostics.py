import unittest
from types import SimpleNamespace
try:
    import torch
except ImportError:
    torch=None
from logit_diagnostics import compare_logits,live_inputs

@unittest.skipIf(torch is None,'CPU torch unavailable in local review interpreter')
class LogitDiagnosticsTests(unittest.TestCase):
    def test_exact_row_permutation(self):
        saved=torch.tensor([[1.,0.,2.],[3.,2.,1.]])
        actual=saved.flip(0)
        d=compare_logits(actual,saved,torch)
        self.assertFalse(d['bitwise']);self.assertFalse(d['argmax_equal'])
        self.assertTrue(d['exact_row_multiset_equal'])
        self.assertEqual([r['exact_saved_row_candidates'] for r in d['rows']],[[1],[0]])
    def test_small_argmax_change_is_preserved(self):
        saved=torch.tensor([[1.,1.00001]])
        actual=saved.flip(1)
        d=compare_logits(actual,saved,torch)
        self.assertFalse(d['argmax_equal']);self.assertGreater(d['max_abs'],0)
        self.assertEqual(d['changed_elements'],2)
        self.assertGreater(d['rows'][0]['saved_top1_margin'],0)
    def test_live_input_identity_and_static_slice(self):
        batch=SimpleNamespace(input_ids=torch.tensor([9,8]),rids=['x','y'])
        buffers=SimpleNamespace(input_ids=torch.tensor([9,8,0,0]),positions=torch.tensor([64,64,0,0]))
        d=live_inputs(batch,SimpleNamespace(buffers=buffers),2,torch)
        self.assertEqual(d['forward_batch']['rids'],['x','y'])
        self.assertEqual(d['captured_input_buffers']['positions']['values'],[64,64])

if __name__=='__main__':unittest.main()
