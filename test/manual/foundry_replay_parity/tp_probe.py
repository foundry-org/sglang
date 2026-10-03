"""CPU-only common admission for TP1/TP2 Foundry replay probes.

Every rank gathers an offer on every backend boundary, before early return or
any additional GPU operation. DP2 uses the more specific dp_probe protocol.
"""

def decide(offers):
    def answer(action, reason):
        return {'protocol':'foundry-tp-probe-v1','action':action,'reason':reason}
    if not isinstance(offers,list) or not offers:
        return answer('reject','invalid_rank_set')
    if any(not isinstance(x,dict) for x in offers):
        return answer('reject','malformed_offer')
    try:
        n=len(offers)
        if sorted(x['rank'] for x in offers)!=list(range(n)):
            return answer('reject','invalid_rank_set')
        if any(x['world_size']!=n for x in offers):
            return answer('reject','incomplete_rank_set')
        if any(x.get('error') for x in offers):
            return answer('reject','local_metadata_error')
        first=offers[0]
        if any(x['replay_index']!=first['replay_index'] for x in offers):
            return answer('reject','boundary_disagreement')
        if any(x['phase']!=first['phase'] for x in offers):
            return answer('skip','phase_publication_race')
        phase=first['phase']
        if type(phase.get('armed')) is not bool:
            return answer('reject','invalid_phase')
        if not phase['armed']:
            return answer('skip','unarmed')
        if type(phase.get('id')) is not int or type(phase.get('batch')) is not int or phase['batch']<1:
            return answer('reject','invalid_phase')
        if any(x['seen'] for x in offers):
            if not all(x['seen'] for x in offers):
                return answer('reject','partially_consumed_phase')
            return answer('skip','already_probed')
        for field in ['runner_name','forward_mode','raw_batch','capture_batch']:
            if any(x[field]!=first[field] for x in offers):
                return answer('reject','rank_metadata_disagreement')
        if first['runner_name']!='DecodeCudaGraphRunner' or first['forward_mode']!='DECODE':
            return answer('skip','not_decode')
        if first['raw_batch']!=phase['batch'] or first['capture_batch']!=phase['batch']:
            return answer('skip','not_requested_exact_batch')
        return answer('probe','all_ranks_agree')
    except (KeyError,TypeError,ValueError):
        return answer('reject','malformed_offer')
