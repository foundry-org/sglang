import json
import tempfile
import unittest
from pathlib import Path
from run_foundry_integration import manifest_groups, complete_pair_sequence

class ManifestSequenceTests(unittest.TestCase):
    def test_actual_members_not_assumed_adjacent_batch(self):
        groups = [{'members':[64,31,30]}, {'members':[32,16]}, {'members':[8]}]
        sequence = complete_pair_sequence(groups)
        remaining = list(sequence)
        for group in groups:
            members = group['members']
            count = len(members)*(len(members)-1)+1
            walk, remaining = remaining[:count],remaining[count:]
            self.assertEqual(walk[0], members[0])
            self.assertEqual(walk[-1], members[0])
            self.assertEqual(set(zip(walk,walk[1:])), {(a,b) for a in members for b in members if a!=b})
        self.assertFalse(remaining)
    def test_singleton_does_not_claim_update(self):
        self.assertEqual(complete_pair_sequence([{'members':[128]}]),[128])
    def test_reads_and_hashes_real_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rank_0';path.mkdir()
            data = {'topology_groups':[{'template':'graph_0_FULL_t31_r31_UX_pcN.json','members':['graph_0_FULL_t31_r31_UX_pcN.json','graph_1_FULL_t30_r30_UX_pcN.json'],'topology_key':'exact','partition':'decode'}]}
            (path/'graph_manifest.json').write_text(json.dumps(data))
            rows=manifest_groups(folder)
            self.assertEqual(rows[0]['groups'][0]['members'],[31,30])
            self.assertEqual(len(rows[0]['sha256']),64)
    def test_rank_group_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for rank,batch in enumerate([30,31]):
                path=Path(folder)/f'rank_{rank}';path.mkdir()
                name=f'graph_0_FULL_t{batch}_r{batch}_UX_pcN.json'
                (path/'graph_manifest.json').write_text(json.dumps({'topology_groups':[{'template':name,'members':[name],'topology_key':'x'}]}))
            with self.assertRaisesRegex(RuntimeError,'different SAVE groups'):
                manifest_groups(folder)
    def test_no_manifest_is_not_a_native_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'No actual SAVE'):
                manifest_groups(folder)

if __name__=='__main__':
    unittest.main()
