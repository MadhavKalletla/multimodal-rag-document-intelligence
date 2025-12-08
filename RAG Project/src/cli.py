# src/cli.py
import argparse
from .agent import answer_question

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str, help="Your question")
    p.add_argument("--k", type=int, default=4, help="top-k passages")
    args = p.parse_args()

    ans, sources = answer_question(args.question, k=args.k)
    print("\n=== ANSWER ===")
    print(ans)
    print("\n=== SOURCES ===")
    for i, s in enumerate(sources, 1):
        src = s.get("meta", {}).get("source", s.get("id", f"chunk_{i}"))
        print(f"[{i}] {src}")

if __name__ == "__main__":
    main()
