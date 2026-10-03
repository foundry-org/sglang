import copy
import unittest
from tp_probe import decide

class TPAdmissionTests(unittest.TestCase):
    def offers(self,n=2):
        return [{'rank':r,'world_size':n,'replay_index':3,'phase':{'armed':True,'id':0,'batch':31},
                 'seen':False,'runner_name':'DecodeCudaGraphRunner','forward_mode':'DECODE',
                 'raw_batch':31,'capture_batch':31,'error':None} for r in range(n)]
    def test_tp1_and_tp2(self):
        for n in (1,2):self.assertEqual(decide(self.offers(n))['action'],'probe')
    def test_phase_race_skips_all(self):
        rows=self.offers();rows[1]['phase']={'armed':False}
        self.assertEqual(decide(rows)['reason'],'phase_publication_race')
    def test_partial_seen_rejects(self):
        rows=self.offers();rows[1]['seen']=True
        self.assertEqual(decide(rows)['action'],'reject')
    def test_batch_disagreement_rejects(self):
        rows=self.offers();rows[1]['raw_batch']=30
        self.assertEqual(decide(rows)['reason'],'rank_metadata_disagreement')
    def test_both_other_batch_skip(self):
        rows=self.offers()
        for row in rows:row['raw_batch']=30
        self.assertEqual(decide(rows)['action'],'skip')
    def test_error_rejects_all(self):
        rows=self.offers();rows[1]['error']='FileNotFoundError'
        self.assertEqual(decide(rows)['action'],'reject')
    def test_unarmed(self):
        rows=self.offers()
        for row in rows:row['phase']={'armed':False}
        self.assertEqual(decide(rows)['reason'],'unarmed')
    def test_counter_disagreement(self):
        rows=self.offers();rows[1]['replay_index']=4
        self.assertEqual(decide(rows)['reason'],'boundary_disagreement')

if __name__=='__main__':unittest.main()
