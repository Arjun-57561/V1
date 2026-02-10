"""
Example Usage of Uncertainty-First Agent Council
Demonstrates various query scenarios and expected behaviors
"""

from uncertainty_agent_council import UncertaintyFirstAgentCouncil
import json


def print_separator(title=""):
    """Print a visual separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def example_1_high_clarity():
    """Example 1: High clarity query with most information present"""
    
    print_separator("EXAMPLE 1: HIGH CLARITY QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    I am 23 years old from Karnataka, earning 2.5 LPA annually. 
    I belong to SC category and work as a farmer. I have Aadhaar card 
    and ration card. Am I eligible for PM-KISAN Samman Nidhi Yojana?
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Moderate-high confidence (~60-70%)
    # Few unknowns, some assumptions
    # Safety: Answer with caution or safe


def example_2_medium_clarity():
    """Example 2: Medium clarity query with some missing info"""
    
    print_separator("EXAMPLE 2: MEDIUM CLARITY QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    I am around 30 years old and earn approximately 25000 per month. 
    Can I get some scholarship for my studies?
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Low-moderate confidence (~40-50%)
    # Several unknowns (state, category, specific scheme)
    # Safety: Answer with caution


def example_3_low_clarity():
    """Example 3: Low clarity query with minimal information"""
    
    print_separator("EXAMPLE 3: LOW CLARITY QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = "Am I eligible for any government schemes?"
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Very low confidence (~10-20%)
    # Many critical unknowns
    # Safety: Unsafe to answer


def example_4_time_sensitive():
    """Example 4: Time-sensitive query requiring current information"""
    
    print_separator("EXAMPLE 4: TIME-SENSITIVE QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    What are the current income tax benefits for senior citizens in 2025? 
    I am 65 years old with pension income of 5 lakhs per year.
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Moderate confidence but HIGH time sensitivity penalty
    # System will warn about potentially outdated information
    # Safety: Answer with caution


def example_5_assumption_heavy():
    """Example 5: Query requiring many assumptions"""
    
    print_separator("EXAMPLE 5: ASSUMPTION-HEAVY QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    I belong to OBC category and my annual income is 4 LPA. 
    Can I apply for educational loans with subsidized interest?
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Moderate confidence but HIGH assumption risk
    # Assumptions: OBC certificate exists, income documented, etc.
    # Safety: Answer with caution


def example_6_legal_domain():
    """Example 6: Legal domain query"""
    
    print_separator("EXAMPLE 6: LEGAL DOMAIN QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    I want to file a consumer complaint against a mobile company. 
    They sold me a defective phone 3 months ago. What are my rights?
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Moderate confidence
    # Missing: State, exact timeline, documentation
    # Domain: legal_prescreening


def example_7_financial_domain():
    """Example 7: Financial compliance domain"""
    
    print_separator("EXAMPLE 7: FINANCIAL DOMAIN QUERY")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    My annual income is 12 lakhs. Do I need to file ITR? 
    What documents do I need?
    """
    
    print(f"Query: {query.strip()}")
    
    response = council.get_user_response(query)
    print(response)
    
    # Expected: Moderate confidence
    # Missing: Income source, deductions, financial year
    # Domain: financial_compliance


def example_8_detailed_extraction():
    """Example 8: Detailed agent output extraction"""
    
    print_separator("EXAMPLE 8: DETAILED AGENT OUTPUT EXTRACTION")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = "I am 28 years old from Delhi with 6 LPA income. Any housing schemes?"
    
    print(f"Query: {query.strip()}\n")
    
    # Get full results
    results = council.process_query(query, verbose=False)
    
    # Extract and display individual agent outputs
    print("\n--- QUERY PROCESSOR OUTPUT ---")
    print(f"Detected Domain: {results['query_processor'].detected_domain}")
    print(f"Entities Extracted:")
    entities = results['query_processor'].entities
    print(f"  - Age: {entities.age}")
    print(f"  - Income: ₹{entities.annual_income:,.0f}" if entities.annual_income else "  - Income: None")
    print(f"  - State: {entities.state_or_ut}")
    print(f"  - Category: {entities.category}")
    
    print("\n--- FACT BOUNDARY OUTPUT ---")
    print(f"Clarity Score: {results['fact_boundary'].clarity_score}")
    print(f"Known Facts: {len(results['fact_boundary'].known_facts)}")
    for fact in results['fact_boundary'].known_facts:
        print(f"  - {fact.description}")
    
    print("\n--- ASSUMPTION AGENT OUTPUT ---")
    print(f"Overall Risk: {results['assumption_agent'].overall_assumption_risk}")
    print(f"Assumptions Made: {len(results['assumption_agent'].assumptions)}")
    for assumption in results['assumption_agent'].assumptions[:3]:
        print(f"  - [{assumption.risk_level.upper()}] {assumption.description}")
    
    print("\n--- UNKNOWN DETECTION OUTPUT ---")
    print(f"Completeness Score: {results['unknown_detection'].information_completeness_score}")
    print(f"Missing Information: {len(results['unknown_detection'].missing_information)}")
    for missing in results['unknown_detection'].missing_information[:3]:
        print(f"  - [{missing.importance_level.upper()}] {missing.field_name}")
    
    print("\n--- TEMPORAL UNCERTAINTY OUTPUT ---")
    print(f"Time Sensitivity: {results['temporal_uncertainty'].time_sensitivity_level}")
    print(f"Time Factors: {len(results['temporal_uncertainty'].time_dependent_factors)}")
    
    print("\n--- CONFIDENCE CALIBRATION OUTPUT ---")
    print(f"Calibrated Confidence: {results['confidence_calibration'].calibrated_confidence}%")
    print(f"Safety Flag: {results['confidence_calibration'].safety_flag}")
    print(f"Explanation: {results['confidence_calibration'].confidence_explanation}")
    
    print("\n--- DECISION GUIDANCE OUTPUT ---")
    print(f"Answer Style: {results['decision_guidance'].final_answer_style}")
    print(f"Knowns: {len(results['decision_guidance'].explicit_knowns)}")
    print(f"Unknowns: {len(results['decision_guidance'].explicit_unknowns)}")
    print(f"Next Steps: {len(results['decision_guidance'].recommended_next_steps)}")


def example_9_batch_processing():
    """Example 9: Batch processing multiple queries"""
    
    print_separator("EXAMPLE 9: BATCH PROCESSING")
    
    council = UncertaintyFirstAgentCouncil()
    
    queries = [
        "I am 22 years old. Any schemes for me?",
        "Monthly income 15000, state Maharashtra, age 35. Eligible for housing loan?",
        "OBC category, 3 LPA income, need scholarship for MBA",
        "Senior citizen tax benefits?",
    ]
    
    results_summary = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query[:50]}... ---")
        
        results = council.process_query(query, verbose=False)
        
        summary = {
            "query": query,
            "confidence": results['confidence_calibration'].calibrated_confidence,
            "safety_flag": results['confidence_calibration'].safety_flag,
            "knowns": len(results['decision_guidance'].explicit_knowns),
            "unknowns": len(results['decision_guidance'].explicit_unknowns),
            "assumptions": len(results['assumption_agent'].assumptions),
        }
        
        results_summary.append(summary)
        
        print(f"Confidence: {summary['confidence']}%")
        print(f"Safety: {summary['safety_flag']}")
        print(f"Knowns: {summary['knowns']} | Unknowns: {summary['unknowns']} | Assumptions: {summary['assumptions']}")
    
    # Summary table
    print("\n" + "="*80)
    print("BATCH SUMMARY")
    print("="*80)
    print(f"{'Query':<50} {'Conf':<6} {'Safety':<20}")
    print("-"*80)
    for s in results_summary:
        query_preview = s['query'][:47] + "..." if len(s['query']) > 50 else s['query']
        print(f"{query_preview:<50} {s['confidence']:<6} {s['safety_flag']:<20}")


def example_10_json_export():
    """Example 10: Exporting results as JSON"""
    
    print_separator("EXAMPLE 10: JSON EXPORT")
    
    council = UncertaintyFirstAgentCouncil()
    
    query = "I am 27, OBC, 4 LPA, from Gujarat. PM Vishwakarma Yojana eligible?"
    
    results = council.process_query(query, verbose=False)
    
    # Convert to JSON-serializable format
    json_output = {
        "user_query": query,
        "query_processor": {
            "cleaned_query": results['query_processor'].cleaned_query,
            "detected_domain": results['query_processor'].detected_domain,
            "entities": {
                "age": results['query_processor'].entities.age,
                "annual_income": results['query_processor'].entities.annual_income,
                "state_or_ut": results['query_processor'].entities.state_or_ut,
                "category": results['query_processor'].entities.category,
            },
            "ambiguity_flags": results['query_processor'].ambiguity_flags,
        },
        "confidence_calibration": {
            "calibrated_confidence": results['confidence_calibration'].calibrated_confidence,
            "safety_flag": results['confidence_calibration'].safety_flag,
        },
        "decision_guidance": {
            "final_answer_style": results['decision_guidance'].final_answer_style,
            "explicit_knowns": results['decision_guidance'].explicit_knowns,
            "explicit_unknowns": results['decision_guidance'].explicit_unknowns,
            "recommended_next_steps": results['decision_guidance'].recommended_next_steps,
        }
    }
    
    # Print formatted JSON
    print(json.dumps(json_output, indent=2))
    
    # Optionally save to file
    # with open('query_result.json', 'w') as f:
    #     json.dump(json_output, f, indent=2)


def run_all_examples():
    """Run all examples"""
    
    examples = [
        ("High Clarity Query", example_1_high_clarity),
        ("Medium Clarity Query", example_2_medium_clarity),
        ("Low Clarity Query", example_3_low_clarity),
        ("Time-Sensitive Query", example_4_time_sensitive),
        ("Assumption-Heavy Query", example_5_assumption_heavy),
        ("Legal Domain Query", example_6_legal_domain),
        ("Financial Domain Query", example_7_financial_domain),
        ("Detailed Output Extraction", example_8_detailed_extraction),
        ("Batch Processing", example_9_batch_processing),
        ("JSON Export", example_10_json_export),
    ]
    
    print("\n" + "#"*80)
    print("  UNCERTAINTY-FIRST AGENT COUNCIL - EXAMPLE SCENARIOS")
    print("#"*80)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"  EXAMPLE {i}: {name}")
        print(f"{'='*80}")
        
        try:
            func()
        except Exception as e:
            print(f"ERROR in {name}: {e}")
        
        if i < len(examples):
            input("\n>>> Press Enter to continue to next example...")
    
    print("\n" + "#"*80)
    print("  ALL EXAMPLES COMPLETED")
    print("#"*80)


if __name__ == "__main__":
    # Run specific example
    # example_1_high_clarity()
    # example_2_medium_clarity()
    # example_3_low_clarity()
    # example_8_detailed_extraction()
    
    # Or run all examples interactively
    run_all_examples()
