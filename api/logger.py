import json
import time
from datetime import datetime

def log_query(query, top_k, retrieved, latency_ms):
    record = {
        "query": query,
        "top_k": top_k,
        "retrieved_chunks": retrieved,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat()
    }

    print(json.dumps(record))
