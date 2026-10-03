import unittest
from types import SimpleNamespace
try:
    import torch
except ImportError:
    torch=None
from logit_diagnostics import compare_logits,live_inputs,correlate_saved_tokens

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

class SavedInputCorrelationTests(unittest.TestCase):
    def test_row_order_uses_request_id(self):
        inputs={'forward_batch':{'rids':['b','a']},'captured_input_buffers':{'input_ids':{'values':[20,10]},'positions':{'values':[64,64]}}}
        result=correlate_saved_tokens(inputs,{'request_ids':['a','b']},{'tokens':[[10,11],[20,21]]},64)
        self.assertTrue(result['mapping_available'])
        self.assertEqual([r['request_index'] for r in result['rows']],[1,0])
        self.assertTrue(result['all_inputs_match_save_at_position'])
        self.assertFalse(result['saved_reference_live_state_known'])
    def test_later_decode_is_not_first_decode(self):
        inputs={'forward_batch':{'rids':['a']},'captured_input_buffers':{'input_ids':{'values':[11]},'positions':{'values':[65]}}}
        result=correlate_saved_tokens(inputs,{'request_ids':['a']},{'tokens':[[10,11]]},64)
        self.assertFalse(result['all_rows_at_first_decode'])
        self.assertTrue(result['all_inputs_match_save_at_position'])
    def test_unknown_rid_has_no_row_order_fallback(self):
        inputs={'forward_batch':{'rids':['other']},'captured_input_buffers':{'input_ids':{'values':[10]},'positions':{'values':[64]}}}
        result=correlate_saved_tokens(inputs,{'request_ids':['a']},{'tokens':[[10,11]]},64)
        self.assertFalse(result['mapping_available'])
        self.assertFalse(result['all_inputs_match_save_at_position'])

if __name__=='__main__':unittest.main()
