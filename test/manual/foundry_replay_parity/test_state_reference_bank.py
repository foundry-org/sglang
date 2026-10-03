import copy
import json
import tempfile
from pathlib import Path
import unittest
from state_reference_bank import (semantic_rows,row_key,common_lookup,add_entry,verify_load_prefixes,digest,global_state_id,Bank,seal,file_sha,validate_global_rows)

class StateBankContracts(unittest.TestCase):
    def setUp(self):
        self.phase={'id':0,'batch':2,'global_batch':2,'request_ids':['a','b']}
        self.generation={'input_ids':[[10,11],[12,13]],'tokens':[[20,21,22],[30,31,32]],
                         'exact_match':True,'independent_save_exact_match':True}
        self.inputs={'forward_batch':{'rids':['b','a']},'captured_input_buffers':{
            'input_ids':{'values':[30,20]},'positions':{'values':[2,2]},'seq_lens':{'values':[3,3]}}}
        for name in ('input_ids','positions','seq_lens'):
            self.inputs['forward_batch'][name]=self.inputs['captured_input_buffers'][name]
    def rows(self): return semantic_rows(self.inputs,self.phase,self.generation,0,2)
    def test_first_decode_consumes_first_generated_only(self):
        row=self.rows()[0]
        self.assertEqual(row['request_index'],1)
        self.assertEqual(row['consumed_prefix'],[12,13,30])
        self.assertEqual(row['expected_next_token'],31)
    def test_later_decode_and_prompt_length_parameterized(self):
        self.inputs['captured_input_buffers']['input_ids']['values']=[31,21]
        self.inputs['captured_input_buffers']['positions']['values']=[3,3]
        self.inputs['captured_input_buffers']['seq_lens']['values']=[4,4]
        self.assertEqual(self.rows()[0]['consumed_prefix'],[12,13,30,31])
    def test_last_decode_has_no_future_reference(self):
        self.inputs['captured_input_buffers']['input_ids']['values']=[32,22]
        self.inputs['captured_input_buffers']['positions']['values']=[4,4]
        self.inputs['captured_input_buffers']['seq_lens']['values']=[5,5]
        self.assertIsNone(self.rows()[0]['expected_next_token'])
    def test_bad_input_rejected(self):
        self.inputs['captured_input_buffers']['input_ids']['values'][0]=99
        with self.assertRaises(ValueError): self.rows()
    def test_bad_position_rejected(self):
        self.inputs['captured_input_buffers']['positions']['values'][0]=1
        with self.assertRaises(ValueError): self.rows()
    def test_bad_length_rejected(self):
        self.inputs['captured_input_buffers']['seq_lens']['values'][0]=4
        with self.assertRaises(ValueError): self.rows()
    def test_unknown_request_rejected(self):
        self.inputs['forward_batch']['rids'][0]='x'
        with self.assertRaises(ValueError): self.rows()
    def test_duplicate_request_rejected(self):
        self.inputs['forward_batch']['rids']=['a','a']
        with self.assertRaises(ValueError): self.rows()
    def test_partial_capture_rejected(self):
        with self.assertRaises(ValueError): semantic_rows(self.inputs,self.phase,self.generation,0,3)
    def test_output_token_not_part_of_input_key(self):
        row=self.rows()[0]; changed=dict(row,expected_next_token=99)
        self.assertEqual(row_key(row),row_key(changed))
    def test_prefix_rank_capture_global_batch_change_key(self):
        row=self.rows()[0]
        for field,value in [('rank',1),('capture_batch',4),('global_batch',4),('consumed_prefix',[99,13,30])]:
            self.assertNotEqual(row_key(row),row_key(dict(row,**{field:value})))
    def test_missing_any_rank_defers(self):
        self.assertEqual(common_lookup([{'available':True},{'available':False}]),'defer')
        self.assertEqual(common_lookup([{'available':True},{'available':True}]),'probe')
        self.assertEqual(common_lookup([{'available':True},{'error':'bad input'}]),'reject')
    def test_conflicting_duplicate_never_promoted(self):
        entry={'semantic':self.rows()[0],'row_sha256':'x','observations':[{'global':'one'}]}
        index={};add_entry(index,'k',copy.deepcopy(entry))
        entry['observations']=[{'global':'two'}];add_entry(index,'k',copy.deepcopy(entry))
        self.assertEqual(len(index['k']['observations']),2)
        entry['row_sha256']='changed'
        with self.assertRaises(ValueError):add_entry(index,'k',entry)
    def test_full_actual_generation_prefix_required(self):
        evidence={'rows':self.rows(),'live_inputs':self.inputs,'actual_complete_prefix_verified':False}
        report={'phase':self.phase,'rank':0,'state_reference_bank':evidence}
        verify_load_prefixes([report],[self.generation]);self.assertTrue(evidence['actual_complete_prefix_verified'])
        bad=copy.deepcopy(self.generation);bad['input_ids'][1][0]=99
        with self.assertRaises(ValueError):verify_load_prefixes([report],[bad])
    def test_live_graph_input_disagreement_rejects(self):
        self.inputs['forward_batch']['input_ids']={'values':[99,20]}
        with self.assertRaises(ValueError): self.rows()
    def test_dp_global_coverage_rejects_duplicate_requests(self):
        rows=self.rows()
        for row in rows: row['global_batch']=4
        second=[dict(row,rank=1) for row in rows]
        with self.assertRaises(ValueError):validate_global_rows([rows,second])
        for row in second:row['request_index']+=2
        self.assertEqual(validate_global_rows([rows,second]),'partitioned_dp')
    def test_tp_global_coverage_requires_complete_replicas(self):
        rows=self.rows();second=[dict(row,rank=1) for row in rows]
        self.assertEqual(validate_global_rows([rows,second]),'replicated_tp')
        second[0]['request_index']=0
        with self.assertRaises(ValueError):validate_global_rows([rows,second])
    def test_global_state_excludes_future_output(self):
        rows=self.rows(); changed=copy.deepcopy(rows);changed[0]['expected_next_token']=99
        self.assertEqual(global_state_id([rows]),global_state_id([changed]))
    def test_global_final_prefix_mismatch_rejected(self):
        rows=self.rows(); evidence={'rows':rows,'live_inputs':self.inputs,
            'global_inputs':[self.inputs],'global_logical_state_id':'wrong'}
        with self.assertRaises(ValueError):
            verify_load_prefixes([{'phase':self.phase,'rank':0,'state_reference_bank':evidence}],[self.generation])
    def test_unsealed_bank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                Bank({'state_reference_bank':d,'mode':'load'},0,None)
    def test_seal_requires_natural_next_token_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'rank_0';folder.mkdir(); tensor=folder/'tensor.pt';tensor.write_bytes(b'retained tensor')
            frame={'ordinal':0,'natural_replay_index':5,'phase':self.phase,'inputs':self.inputs,
                'global_inputs':[self.inputs],'tensor':'tensor.pt','file_sha256':file_sha(tensor),
                'row_sha256':['a','b'],'argmax':[31,21]}
            (folder/'frames.jsonl').write_text(json.dumps(frame)+'\n')
            generation={**self.generation,'phase':self.phase}
            result=seal(d,[generation],{'capture_batches':[2]},[{'sha256':'archive'}],1,None)
            self.assertTrue(result['sealed_after_generation_validation'])
            self.assertEqual(result['summary']['0']['unique_keys'],2)
            (Path(d)/'sealed.json').unlink()
            frame['argmax'][0]=99
            (folder/'frames.jsonl').write_text(json.dumps(frame)+'\n')
            with self.assertRaises(ValueError):seal(d,[generation],{'capture_batches':[2]},[{'sha256':'archive'}],1,None)
            self.assertFalse((Path(d)/'sealed.json').exists())
    def test_failed_final_generation_cannot_pass(self):
        report={'phase':self.phase,'rank':0,'state_reference_bank':{'rows':self.rows(),'live_inputs':self.inputs}}
        self.generation['independent_save_exact_match']=False
        with self.assertRaises(ValueError):verify_load_prefixes([report],[self.generation])

if __name__=='__main__': unittest.main()
