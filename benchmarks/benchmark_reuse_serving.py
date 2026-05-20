"""
FlexKV KV Cache Reuse Benchmark for Online Serving.

Sends requests to a vLLM OpenAI-compatible chat completions endpoint using
patterns that produce prefix-level KV cache reuse:

  Pattern 1 – Shared System Prompt
    All requests carry the same long system message (>256 tokens) plus a
    different user question.  The system-prompt KV is stored on the first
    request and reloaded (H2D) for every subsequent one.

  Flush  – CPU Cache Eviction (only when --test-ssd is set)
    Sends many different large prompts to fill the CPU cache, forcing earlier
    KV data to be evicted to SSD.

  Pattern 2 – Multi-turn Conversation
    Picks conversations from a ShareGPT JSON file and replays them turn by
    turn, accumulating history.  Each new turn shares the full prefix of all
    prior turns, so FlexKV should match a growing prefix.

  Pattern 3 – Shared System Prompt Again (only when --test-ssd is set)
    Re-sends the same system-prompt requests from Pattern 1.  Since the KV
    was evicted from CPU to SSD during the flush phase, these requests must
    trigger SSD→CPU (DISK2H) + CPU→GPU (H2D) transfers.

Usage:
    # CPU-only reuse test:
    python benchmark_reuse_serving.py \
        --model /workspace/gemma-4-31B-it \
        --dataset-path /workspace/ShareGPT_V3_unfiltered_cleaned_split.json

    # SSD reuse test (adds flush + pattern 3):
    python benchmark_reuse_serving.py --test-ssd \
        --model /workspace/gemma-4-31B-it \
        --dataset-path /workspace/ShareGPT_V3_unfiltered_cleaned_split.json
"""

import argparse
import asyncio
import json
import time
import aiohttp


# ---------------------------------------------------------------------------
# Long system prompt (~350+ tokens after tokenization) used by Pattern 1.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in software engineering, "
    "distributed systems, and machine learning infrastructure. You have deep "
    "knowledge of GPU programming with CUDA, memory management, KV cache "
    "optimization, and large language model serving systems. When answering "
    "questions, you should provide detailed technical explanations with "
    "concrete examples. Always consider performance implications, memory "
    "usage, and scalability. If a question involves code, provide working "
    "code snippets with proper error handling. You should think step by step "
    "and break down complex problems into manageable parts. Your responses "
    "should be accurate, concise, and actionable. When discussing trade-offs, "
    "present multiple options with their pros and cons. You are also familiar "
    "with popular frameworks like PyTorch, vLLM, TensorRT-LLM, and various "
    "attention mechanisms including Multi-Head Attention, Grouped-Query "
    "Attention, and Multi-Latent Attention. You understand the internals of "
    "transformer architectures, including how KV caches work, how prefix "
    "caching enables sharing computation across requests, and how paged "
    "attention manages memory efficiently. You can explain concepts at "
    "different levels of abstraction depending on the audience."
)

USER_QUESTIONS = [
    "What is the difference between paged attention and flash attention?",
    "How does KV cache eviction work in a typical LLM serving system?",
    "Explain how tensor parallelism splits the attention computation across GPUs.",
    "What are the main bottlenecks in LLM inference and how can they be addressed?",
    "How does speculative decoding improve inference throughput?",
    "What is the role of a radix tree in prefix caching?",
    "Compare continuous batching vs static batching for LLM serving.",
    "How does grouped-query attention reduce memory usage compared to multi-head attention?",
    "Explain the concept of pipeline parallelism in distributed model serving.",
    "What are the trade-offs between FP16 and INT8 quantization for KV caches?",
    "How does chunked prefill help balance prefill and decode phases?",
    "What is the purpose of a block manager in vLLM's memory management?",
    "How can you efficiently transfer KV cache data between CPU and GPU memory?",
    "What are the challenges of serving models with heterogeneous attention patterns?",
    "Explain how ring attention enables processing very long sequences.",
    "What is the difference between data parallelism and tensor parallelism?",
    "How does CUDA memory pooling improve allocation performance?",
    "What strategies can reduce time-to-first-token in LLM serving?",
    "How does prefix caching interact with beam search decoding?",
    "Explain the memory layout considerations for KV cache on GPU vs CPU.",
]


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    request_label: str,
) -> dict:
    """Send a single chat completion request and return the result."""
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.json()
            elapsed = time.perf_counter() - t0
            if resp.status != 200:
                error_msg = body.get("error", {}).get("message", str(body))
                print(f"  [{request_label}] ERROR {resp.status}: {error_msg}")
                return {"success": False, "label": request_label, "error": error_msg}
            content = body["choices"][0]["message"]["content"]
            print(f"  [{request_label}] {elapsed:.2f}s | {content[:80]}")
            return {"success": True, "label": request_label, "content": content}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{request_label}] EXCEPTION after {elapsed:.2f}s: {e}")
        return {"success": False, "label": request_label, "error": str(e)}


# Distinct long prompts used to flush CPU cache and force eviction to SSD.
FLUSH_PROMPTS = [
    "Write a comprehensive essay about the history of the Roman Empire from its founding in 753 BC through the Republic period, the Punic Wars against Carthage, the rise of Julius Caesar, the transition to Empire under Augustus, the Pax Romana, the Crisis of the Third Century, the division into Eastern and Western empires, and finally the fall of the Western Roman Empire in 476 AD. Include discussion of key figures, military campaigns, political institutions, cultural achievements, and the lasting legacy of Rome on Western civilization.",
    "Explain the complete theory of general relativity as formulated by Albert Einstein in 1915. Start with the equivalence principle and the concept of spacetime curvature. Derive the Einstein field equations and explain each tensor component. Discuss the Schwarzschild solution and its implications for black holes. Explain gravitational time dilation, gravitational lensing, and gravitational waves. Compare general relativity with Newtonian gravity and discuss where they diverge. Finally, explain the current challenges in reconciling general relativity with quantum mechanics.",
    "Provide a detailed guide to modern compiler design and implementation. Cover lexical analysis with finite automata and regular expressions, parsing with context-free grammars and LR/LALR techniques, semantic analysis with type checking and symbol tables, intermediate representation generation including SSA form, optimization passes including constant folding dead code elimination loop optimization and register allocation, and finally code generation for x86-64 and ARM architectures. Include discussion of garbage collection strategies and just-in-time compilation.",
    "Describe the complete biochemistry of human metabolism including glycolysis the citric acid cycle oxidative phosphorylation fatty acid oxidation amino acid metabolism the urea cycle gluconeogenesis glycogenesis and glycogenolysis. Explain the regulation of each pathway through allosteric enzymes hormonal control and substrate availability. Discuss the integration of metabolism across different organs including the liver muscles brain and adipose tissue. Include the role of vitamins and cofactors in enzymatic reactions.",
    "Write a thorough analysis of distributed consensus algorithms used in modern computing systems. Cover the FLP impossibility result and its implications. Explain Paxos in detail including Multi-Paxos and its optimizations. Compare with Raft and explain why it was designed as a more understandable alternative. Discuss Byzantine fault tolerance with PBFT and newer protocols like HotStuff. Cover the CAP theorem and its practical implications for distributed databases. Explain how systems like Google Spanner achieve external consistency.",
    "Provide a comprehensive overview of quantum computing from fundamental principles to current applications. Explain qubits quantum gates and quantum circuits. Cover major quantum algorithms including Shor algorithm for factoring Grover search algorithm quantum simulation and variational quantum eigensolver. Discuss quantum error correction codes including surface codes and their threshold theorems. Explain different physical implementations including superconducting qubits trapped ions photonic systems and topological qubits. Discuss current limitations and the path to fault-tolerant quantum computing.",
    "Write a detailed exploration of the human immune system covering both innate and adaptive immunity. Explain the role of physical barriers complement system natural killer cells macrophages and dendritic cells in innate immunity. Cover T cell development in the thymus MHC restriction positive and negative selection. Explain B cell maturation antibody structure class switching and affinity maturation. Discuss the generation of immunological memory and how vaccines exploit this mechanism. Cover immunodeficiency disorders autoimmune diseases and current immunotherapy approaches for cancer treatment.",
    "Explain the complete architecture and internals of modern operating systems. Cover process management including scheduling algorithms context switching and inter-process communication. Discuss memory management with virtual memory paging segmentation and page replacement algorithms. Explain file systems including ext4 NTFS and ZFS with their B-tree and copy-on-write designs. Cover device drivers and the I/O subsystem. Discuss security mechanisms including access control lists capabilities and mandatory access control. Finally explain containerization and virtualization technologies.",
    "Provide a thorough analysis of deep learning architectures from foundational neural networks to modern transformers. Cover backpropagation and gradient descent optimization methods including Adam and LAMB. Explain convolutional neural networks and their application in computer vision. Discuss recurrent neural networks LSTMs and their limitations. Cover the transformer architecture in detail including multi-head attention positional encoding and layer normalization. Explain modern variants including GPT BERT T5 and mixture-of-experts models. Discuss training techniques including mixed precision distributed training and curriculum learning.",
    "Write a comprehensive guide to cryptographic systems and protocols. Cover symmetric encryption with AES and ChaCha20. Explain asymmetric cryptography including RSA elliptic curve cryptography and post-quantum lattice-based schemes. Discuss hash functions including SHA-256 and BLAKE3. Cover digital signatures key exchange protocols like Diffie-Hellman and key management systems. Explain TLS 1.3 protocol in detail. Discuss zero-knowledge proofs homomorphic encryption and secure multi-party computation. Cover blockchain consensus mechanisms and their cryptographic foundations.",
]


async def flush_cpu_cache(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Send many different large prompts to fill CPU cache, forcing eviction to SSD."""
    print()
    print("=" * 70)
    print("Flush Phase: Filling CPU cache to force eviction to SSD")
    print(f"  Sending {len(FLUSH_PROMPTS)} large unique prompts...")
    print("=" * 70)

    results = []
    for i, prompt in enumerate(FLUSH_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        result = await send_chat_request(
            session, url, model, messages, max_tokens,
            request_label=f"FLUSH-{i+1:02d}",
        )
        results.append(result)
    return results


async def pattern1_shared_system_prompt(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Pattern 1: all requests share the same long system prompt."""
    print("=" * 70)
    print("Pattern 1: Shared System Prompt (20 requests)")
    print("  System prompt will be cached after first request.")
    print("  Subsequent requests should trigger H2D for shared prefix.")
    print("=" * 70)

    results = []
    for i, question in enumerate(USER_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        result = await send_chat_request(
            session, url, model, messages, max_tokens,
            request_label=f"P1-{i+1:02d}",
        )
        results.append(result)
    return results


def load_multiturn_conversations(dataset_path: str, num_convs: int = 5, min_turns: int = 6):
    """Load multi-turn conversations from ShareGPT dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    selected = []
    for conv in data:
        turns = conv.get("conversations", [])
        # Need at least min_turns human/gpt pairs, starting with human
        if len(turns) >= min_turns and turns[0].get("from") == "human":
            # Verify alternating human/gpt pattern
            valid = True
            for j in range(0, min(len(turns), min_turns)):
                expected = "human" if j % 2 == 0 else "gpt"
                if turns[j].get("from") != expected:
                    valid = False
                    break
            if valid:
                selected.append(conv)
                if len(selected) >= num_convs:
                    break
    return selected


async def pattern2_multiturn_conversation(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    max_tokens: int,
    dataset_path: str,
) -> list[dict]:
    """Pattern 2: replay multi-turn conversations, accumulating history."""
    print()
    print("=" * 70)
    print("Pattern 2: Multi-turn Conversations")
    print("  Each new turn includes all prior turns as prefix.")
    print("  FlexKV should match the growing shared prefix via H2D.")
    print("=" * 70)

    conversations = load_multiturn_conversations(dataset_path, num_convs=5, min_turns=6)
    if not conversations:
        print("  WARNING: No suitable multi-turn conversations found in dataset.")
        return []

    print(f"  Selected {len(conversations)} conversations with >=6 turns.")

    results = []
    for ci, conv in enumerate(conversations):
        turns = conv.get("conversations", [])
        conv_id = conv.get("id", f"conv-{ci}")
        # Use up to 6 turns (3 human turns = 3 requests with growing history)
        max_human_turns = 3
        print(f"\n  --- Conversation {ci+1} (id={conv_id}, total_turns={len(turns)}) ---")

        messages_so_far = []
        human_turn_count = 0
        for ti, turn in enumerate(turns):
            role_from = turn.get("from", "")
            value = turn.get("value", "")

            if role_from == "human":
                messages_so_far.append({"role": "user", "content": value})
                human_turn_count += 1

                # Send request with accumulated messages
                result = await send_chat_request(
                    session, url, model, list(messages_so_far), max_tokens,
                    request_label=f"P2-C{ci+1}-T{human_turn_count}",
                )
                results.append(result)

                # Use the model's actual response as the assistant turn
                if result["success"]:
                    messages_so_far.append({
                        "role": "assistant",
                        "content": result["content"],
                    })
                else:
                    # Use original gpt response as fallback
                    if ti + 1 < len(turns) and turns[ti + 1].get("from") == "gpt":
                        messages_so_far.append({
                            "role": "assistant",
                            "content": turns[ti + 1]["value"],
                        })
                    break

                if human_turn_count >= max_human_turns:
                    break
            elif role_from == "gpt":
                # Skip gpt turns in input — we use the model's own response
                continue

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="FlexKV KV Cache Reuse Benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", default=None,
                        help="Path to ShareGPT JSON (required for Pattern 2)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--test-ssd", action="store_true",
                        help="Enable SSD test mode: adds flush phase + Pattern 3 "
                             "to force and verify SSD->CPU->GPU reload")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"Target: {url}")
    print(f"Model:  {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"SSD test: {args.test_ssd}")

    all_results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Pattern 1: shared system prompt
        r1 = await pattern1_shared_system_prompt(
            session, url, args.model, args.max_tokens)
        all_results.extend(r1)

        if args.test_ssd:
            # Flush: fill CPU cache to push Pattern 1 KV data to SSD
            rf = await flush_cpu_cache(
                session, url, args.model, args.max_tokens)
            all_results.extend(rf)

        # Pattern 2: multi-turn conversations
        if args.dataset_path:
            r2 = await pattern2_multiturn_conversation(
                session, url, args.model, args.max_tokens, args.dataset_path)
            all_results.extend(r2)
        else:
            print("\n  Skipping Pattern 2 (no --dataset-path provided)")

        if args.test_ssd:
            # Pattern 3: re-send system prompt requests — should trigger DISK2H
            print()
            print("=" * 70)
            print("Pattern 3: Re-send System Prompt (after SSD eviction)")
            print("  System prompt KV was evicted to SSD during flush.")
            print("  These requests should trigger DISK2H + H2D.")
            print("=" * 70)
            for i, question in enumerate(USER_QUESTIONS[:5]):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                result = await send_chat_request(
                    session, url, args.model, messages, args.max_tokens,
                    request_label=f"P3-{i+1:02d}",
                )
                all_results.append(result)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    failures = total - successes
    print(f"  Total requests:  {total}")
    print(f"  Successes:       {successes}")
    print(f"  Failures:        {failures}")
    if failures > 0:
        print("  Failed requests:")
        for r in all_results:
            if not r["success"]:
                print(f"    [{r['label']}] {r.get('error', 'unknown')}")
    print()
    if failures == 0:
        print("  All requests completed successfully.")
    else:
        print(f"  WARNING: {failures} request(s) failed!")


if __name__ == "__main__":
    asyncio.run(main())
