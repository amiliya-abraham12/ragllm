"""
RAG Evaluation Module
- Retrieval precision metrics
- Answer faithfulness scoring
- Latency tracking
"""

import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Single evaluation result"""
    query: str
    precision: float
    hits: int
    total_expected: int
    latency_ms: float
    answer: Optional[str] = None
    faithfulness: Optional[float] = None


class RAGEvaluator:
    """
    Evaluator for RAG system quality metrics.
    
    Metrics:
    - Retrieval Precision: % of expected keywords found in retrieved chunks
    - Answer Faithfulness: % of answer words grounded in context
    - Latency: Response time in milliseconds
    """
    
    def __init__(self, retriever, chat_fn=None):
        """
        Args:
            retriever: HybridRetriever instance
            chat_fn: Optional ask() function for end-to-end evaluation
        """
        self.retriever = retriever
        self.chat_fn = chat_fn
        self.results: List[EvalResult] = []
    
    def evaluate_retrieval(
        self,
        query: str,
        expected_keywords: List[str],
        top_k: int = 3
    ) -> EvalResult:
        """
        Check if retrieved chunks contain expected keywords.
        
        Args:
            query: Search query
            expected_keywords: Keywords that should appear in results
            top_k: Number of results to retrieve
        """
        start = time.time()
        results = self.retriever.retrieve(query, top_k=top_k)
        latency = (time.time() - start) * 1000
        
        # Combine all retrieved text
        retrieved_text = ' '.join([r.text.lower() for r in results])
        
        # Count keyword hits
        hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
        precision = hits / len(expected_keywords) if expected_keywords else 0.0
        
        return EvalResult(
            query=query,
            precision=precision,
            hits=hits,
            total_expected=len(expected_keywords),
            latency_ms=latency
        )
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure what % of answer content is grounded in context.
        Higher = more faithful to source.
        """
        answer_words = set(
            w.lower() for w in answer.split() 
            if len(w) > 3 and w.isalpha()
        )
        
        if not answer_words:
            return 1.0
        
        context_lower = context.lower()
        grounded = sum(1 for w in answer_words if w in context_lower)
        return grounded / len(answer_words)
    
    def run_test_suite(self, test_cases: List[Dict]) -> Dict:
        """
        Run full evaluation suite.
        
        test_cases format:
        [
            {
                'query': 'What is minimum GPA for transfer?',
                'expected_keywords': ['GPA', '3.0', 'transfer', 'eligibility']
            }
        ]
        """
        results = []
        total_latency = 0
        
        for case in test_cases:
            result = self.evaluate_retrieval(
                case['query'],
                case.get('expected_keywords', [])
            )
            total_latency += result.latency_ms
            results.append(result)
        
        avg_precision = sum(r.precision for r in results) / len(results) if results else 0
        avg_latency = total_latency / len(results) if results else 0
        
        return {
            'avg_precision': avg_precision,
            'avg_latency_ms': avg_latency,
            'test_count': len(results),
            'passed': sum(1 for r in results if r.precision >= 0.5),
            'details': results
        }
    
    def print_report(self, summary: Dict) -> None:
        """Print evaluation report"""
        print("\n" + "=" * 50)
        print("RAG EVALUATION REPORT")
        print("=" * 50)
        print(f"Tests Run: {summary['test_count']}")
        print(f"Passed (>50% precision): {summary['passed']}/{summary['test_count']}")
        print(f"Average Precision: {summary['avg_precision']:.1%}")
        print(f"Average Latency: {summary['avg_latency_ms']:.0f}ms")
        print("=" * 50)


# Sample test cases for university policy Q&A
SAMPLE_TEST_CASES = [
    {
        'id': 'TC001',
        'query': 'What are the eligibility criteria for inter-college transfer?',
        'expected_keywords': ['eligibility', 'transfer', 'credits', 'gpa'],
        'category': 'academic'
    },
    {
        'id': 'TC002',
        'query': 'What is the deadline for fee payment?',
        'expected_keywords': ['deadline', 'fee', 'payment', 'date'],
        'category': 'finance'
    },
    {
        'id': 'TC003',
        'query': 'What is the attendance policy?',
        'expected_keywords': ['attendance', 'minimum', 'percentage'],
        'category': 'academic'
    },
    {
        'id': 'TC004',
        'query': 'How to apply for re-evaluation?',
        'expected_keywords': ['re-evaluation', 'application', 'fee'],
        'category': 'examination'
    },
    {
        'id': 'TC005',
        'query': 'What are the library rules?',
        'expected_keywords': ['library', 'rules', 'borrowing', 'books'],
        'category': 'facilities'
    }
]


def run_quick_eval(retriever) -> Dict:
    """Quick evaluation with sample test cases"""
    evaluator = RAGEvaluator(retriever)
    summary = evaluator.run_test_suite(SAMPLE_TEST_CASES)
    evaluator.print_report(summary)
    return summary


if __name__ == "__main__":
    # Example usage (requires initialized retriever)
    print("RAG Evaluator Module")
    print("Usage: from backend.evaluate import RAGEvaluator, run_quick_eval")
    print("       summary = run_quick_eval(retriever)")
