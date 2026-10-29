import time
from src.bot.retriever import load_models_and_milvus

TEST_DATASET = [
    {
        "question": "What is the penalty for parking in the VIP sector without authorization?",
        "expected_keywords": ["$50", "warning", "vip"]
    },
    {
        "question": "How much does overnight parking cost for employees?",
        "expected_keywords": ["$15", "overnight", "payroll"]
    },
    {
        "question": "How many hours are free for guest parking?",
        "expected_keywords": ["4 hours free", "guest"]
    },
    {
        "question": "Can I reserve a spot in Sector A?",
        "expected_keywords": ["sector a", "first-come"]
    },
    {
        "question": "What happens if I block an EV charging station?",
        "expected_keywords": ["towing", "$150", "blocking"]
    }
]

def evaluate_retrieval():
    llm, retriever = load_models_and_milvus()
    
    k = 3 

    total_latency = 0
    total_recall = 0
    total_precision = 0

    print(f"Starting RAG evaluation (Top-{k} results)...\n")

    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"--- RAG EVALUATION REPORT (Top-{k}) ---\n\n")

        for i, item in enumerate(TEST_DATASET):
            query = item["question"]
            keywords = item["expected_keywords"]

            start_time = time.time()
            docs = retriever.invoke(query)
            latency = time.time() - start_time

            relevant_chunks = 0
            for doc in docs:
                text = doc.page_content.lower()
                if any(kw.lower() in text for kw in keywords):
                    relevant_chunks += 1

            recall = 1 if relevant_chunks > 0 else 0

            precision = relevant_chunks / len(docs) if docs else 0

            total_latency += latency
            total_recall += recall
            total_precision += precision

            report_chunk = (
                f"Test {i+1}: {query}\n"
                f"Latency: {latency:.4f}s\n"
                f"Recall@{k}: {recall}\n"
                f"Precision@{k}: {precision:.2f}\n"
                f"Retrieved Chunks: {len(docs)}\n"
                f"----------------------------------------\n"
            )
            print(report_chunk, end="")
            f.write(report_chunk)

        avg_latency = total_latency / len(TEST_DATASET)
        avg_recall = total_recall / len(TEST_DATASET)
        avg_precision = total_precision / len(TEST_DATASET)

        summary = (
            "\n=== FINAL SUMMARY ===\n"
            f"Average Latency: {avg_latency:.4f}s (Speed)\n"
            f"Average Recall@{k}: {avg_recall:.2f} (Target: 1.0)\n"
            f"Average Precision@{k}: {avg_precision:.2f} (Higher is better)\n"
        )
        print(summary)
        f.write(summary)

    print("\nRAG evaluation completed. Results saved in 'evaluation_report.txt'.")

if __name__ == "__main__":
    evaluate_retrieval()