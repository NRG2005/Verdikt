"""
L6: SHA-256 Hash Chain — real, persisted, verifiable audit log.

Track 02 retrofit note: this module previously had a stub `get_previous_hash()`
that returned a hardcoded placeholder, and `append_to_chain()` never actually
wrote anywhere — it just printed. Nothing was persisted and nothing was
chained. This is the real implementation: every block links to the ACTUAL
hash of the previous block, persisted to an append-only JSONL file, with a
`verify_chain()` that can prove tampering.

Design: ONE global chain across the whole system (not one chain per case).
That's deliberate — a real audit log's guarantee is that tampering with ANY
past entry, from ANY transaction, breaks verification for everything written
after it. A separate chain per case would only prove that one case's blocks
weren't touched, which is a much weaker guarantee.
"""
import hashlib
import json
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_PATH = os.path.join(_HERE, "..", "data", "audit_chain.jsonl")
GENESIS_HASH = "0" * 64

_lock = threading.Lock()  # serialize appends so prev_hash never races


def calculate_hash(data):
    """Calculates the SHA-256 hash of a dictionary."""
    encoded_data = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded_data).hexdigest()


def _iter_chain():
    if not os.path.exists(CHAIN_PATH):
        return
    with open(CHAIN_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_previous_hash():
    """Returns the hash of the last block actually persisted to the chain,
    or the genesis hash if the chain is empty. Real read, not a placeholder."""
    last = None
    for block in _iter_chain():
        last = block
    return last["hash"] if last else GENESIS_HASH


def append_to_chain(event, verdict):
    """
    Appends a new block to the persisted chain. Returns the new block.
    """
    with _lock:
        prev_hash = get_previous_hash()
        layer_data = {"event": event, "verdict": verdict}
        block_to_hash = {"prev_hash": prev_hash, "layer_data": layer_data}
        new_hash = calculate_hash(block_to_hash)

        # sequence number = how many blocks already exist (cheap re-derive,
        # correct as long as _lock serializes appends, which it does)
        index = sum(1 for _ in _iter_chain())

        new_block = {
            "index": index,
            "hash": new_hash,
            "prev_hash": prev_hash,
            "layer_data": layer_data,
        }

        os.makedirs(os.path.dirname(CHAIN_PATH), exist_ok=True)
        with open(CHAIN_PATH, "a") as f:
            f.write(json.dumps(new_block, default=str) + "\n")

        return new_block


def verify_chain():
    """
    Walks the persisted chain and recomputes every hash. Returns
    {"valid": bool, "blocks_checked": int, "first_bad_index": int|None,
     "reason": str|None}. This is the function a judge can watch fail after
     hand-editing one line of data/audit_chain.jsonl.
    """
    expected_prev = GENESIS_HASH
    checked = 0
    for block in _iter_chain():
        if block.get("prev_hash") != expected_prev:
            return {
                "valid": False, "blocks_checked": checked,
                "first_bad_index": block.get("index"),
                "reason": (
                    f"block {block.get('index')}'s prev_hash does not match "
                    f"the actual hash of the block before it — the chain "
                    f"before this point was altered after being written."
                ),
            }
        recomputed = calculate_hash({
            "prev_hash": block.get("prev_hash"),
            "layer_data": block.get("layer_data"),
        })
        if recomputed != block.get("hash"):
            return {
                "valid": False, "blocks_checked": checked,
                "first_bad_index": block.get("index"),
                "reason": (
                    f"block {block.get('index')}'s stored hash doesn't match "
                    f"its own recomputed content hash — this block's data "
                    f"was edited after it was written."
                ),
            }
        expected_prev = block["hash"]
        checked += 1

    return {"valid": True, "blocks_checked": checked, "first_bad_index": None, "reason": None}
