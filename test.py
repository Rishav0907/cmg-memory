
import sys; sys.path.insert(0,'.')
from cmg.store import InMemoryVectorStore
from cmg.consolidation import ConsolidationEngine, _get_sessions_seen, FAST_TRACK_TYPES, PROMOTION_THRESHOLDS
from cmg.types import MemoryLayer, MemoryType

store = InMemoryVectorStore(persist_path='./cmg_memory_store.json')
engine = ConsolidationEngine(store)

for c in store.all_chunks():
    if c.superseded_by: continue
    sessions = _get_sessions_seen(c)
    if c.layer not in PROMOTION_THRESHOLDS: 
        print(f'SKIP (layer {c.layer.value} not in thresholds): {c.content[:40]}')
        continue
    _, _, _, target = PROMOTION_THRESHOLDS[c.layer]
    goal_thresh = 0.45 if c.memory_type == MemoryType.GOAL else 0.72
    ft = (c.layer == MemoryLayer.EPISODIC and target == MemoryLayer.SEMANTIC
          and c.memory_type in FAST_TRACK_TYPES and c.score >= goal_thresh and sessions >= 1)
    print(f'[{c.layer.value}][{c.memory_type.value}] score={c.score:.3f} sess={sessions} ft={ft}')
    print(f'  layer_in_promo={c.layer in PROMOTION_THRESHOLDS} target={target.value if c.layer in PROMOTION_THRESHOLDS else "N/A"}')
    print(f'  {c.content[:50]}')