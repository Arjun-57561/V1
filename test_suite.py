"""
Test Suite for Uncertainty-First Agent Council
Validates all agents and confidence calibration logic
"""

from uncertainty_agent_council import (
    UncertaintyFirstAgentCouncil,
    QueryProcessor,
    FactBoundaryAgent,
    AssumptionAgent,
    UnknownDetectionAgent,
    TemporalUncertaintyAgent,
    ConfidenceCalibrationAgent,
    DecisionGuidanceAgent,
    SafetyFlag,
    RiskLevel,
    Domain
)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_pass(self, test_name):
        self.passed += 1
        self.tests.append((test_name, "PASS"))
        print(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name, reason):
        self.failed += 1
        self.tests.append((test_name, f"FAIL: {reason}"))
        print(f"❌ FAIL: {test_name} - {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({self.passed/total*100:.1f}%)")
        print(f"Failed: {self.failed} ({self.failed/total*100:.1f}%)")
        print("="*80)
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for name, status in self.tests:
                if status.startswith("FAIL"):
                    print(f"  - {name}: {status}")


# =============================================================================
# AGENT 1: QUERY PROCESSOR TESTS
# =============================================================================

def test_query_processor(results: TestResults):
    """Test Query Processor agent"""
    
    print("\n" + "="*80)
    print("TESTING QUERY PROCESSOR")
    print("="*80)
    
    qp = QueryProcessor()
    
    # Test 1: Age extraction
    query = "I am 25 years old"
    output = qp.process(query)
    if output.entities.age == 25:
        results.add_pass("QP: Age extraction")
    else:
        results.add_fail("QP: Age extraction", f"Expected 25, got {output.entities.age}")
    
    # Test 2: Income extraction (LPA)
    query = "My annual income is 3 LPA"
    output = qp.process(query)
    if output.entities.annual_income == 300000:
        results.add_pass("QP: Income extraction (LPA)")
    else:
        results.add_fail("QP: Income extraction (LPA)", 
                        f"Expected 300000, got {output.entities.annual_income}")
    
    # Test 3: State extraction
    query = "I am from Maharashtra"
    output = qp.process(query)
    if output.entities.state_or_ut == "Maharashtra":
        results.add_pass("QP: State extraction")
    else:
        results.add_fail("QP: State extraction", 
                        f"Expected Maharashtra, got {output.entities.state_or_ut}")
    
    # Test 4: Category extraction
    query = "I belong to OBC category"
    output = qp.process(query)
    if output.entities.category == "OBC":
        results.add_pass("QP: Category extraction")
    else:
        results.add_fail("QP: Category extraction", 
                        f"Expected OBC, got {output.entities.category}")
    
    # Test 5: Domain detection - Government scheme
    query = "Am I eligible for PM-KISAN scheme?"
    output = qp.process(query)
    if output.detected_domain == Domain.GOVERNMENT_SCHEME.value:
        results.add_pass("QP: Domain detection (scheme)")
    else:
        results.add_fail("QP: Domain detection (scheme)", 
                        f"Expected government_scheme, got {output.detected_domain}")
    
    # Test 6: Domain detection - Legal
    query = "What are my legal rights in consumer case?"
    output = qp.process(query)
    if output.detected_domain == Domain.LEGAL_PRESCREENING.value:
        results.add_pass("QP: Domain detection (legal)")
    else:
        results.add_fail("QP: Domain detection (legal)", 
                        f"Expected legal_prescreening, got {output.detected_domain}")
    
    # Test 7: Ambiguity detection
    query = "I am maybe around 30 years old"
    output = qp.process(query)
    if len(output.ambiguity_flags) > 0:
        results.add_pass("QP: Ambiguity detection")
    else:
        results.add_fail("QP: Ambiguity detection", "No ambiguity flags found")


# =============================================================================
# AGENT 2: FACT BOUNDARY TESTS
# =============================================================================

def test_fact_boundary_agent(results: TestResults):
    """Test Fact Boundary Agent"""
    
    print("\n" + "="*80)
    print("TESTING FACT BOUNDARY AGENT")
    print("="*80)
    
    qp = QueryProcessor()
    fb = FactBoundaryAgent()
    
    # Test 1: High clarity score with complete info
    query = "I am 25 years old from Maharashtra with 3 LPA income, OBC category"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    
    if fb_output.clarity_score >= 0.6:
        results.add_pass("FB: High clarity score")
    else:
        results.add_fail("FB: High clarity score", 
                        f"Expected >=0.6, got {fb_output.clarity_score}")
    
    # Test 2: Low clarity score with minimal info
    query = "Am I eligible?"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    
    if fb_output.clarity_score <= 0.3:
        results.add_pass("FB: Low clarity score")
    else:
        results.add_fail("FB: Low clarity score", 
                        f"Expected <=0.3, got {fb_output.clarity_score}")
    
    # Test 3: Correct number of facts extracted
    query = "I am 23 years old from Karnataka with 2.5 LPA income"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    
    if len(fb_output.known_facts) >= 3:  # Age, state, income
        results.add_pass("FB: Facts extraction count")
    else:
        results.add_fail("FB: Facts extraction count", 
                        f"Expected >=3 facts, got {len(fb_output.known_facts)}")


# =============================================================================
# AGENT 3: ASSUMPTION AGENT TESTS
# =============================================================================

def test_assumption_agent(results: TestResults):
    """Test Assumption Agent"""
    
    print("\n" + "="*80)
    print("TESTING ASSUMPTION AGENT")
    print("="*80)
    
    qp = QueryProcessor()
    fb = FactBoundaryAgent()
    aa = AssumptionAgent()
    
    # Test 1: High-risk assumptions for scheme queries
    query = "I belong to OBC with 3 LPA income. Am I eligible for schemes?"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    aa_output = aa.process(query, qp_output, fb_output)
    
    high_risk_count = sum(1 for a in aa_output.assumptions 
                         if a.risk_level == RiskLevel.HIGH.value)
    
    if high_risk_count >= 1:
        results.add_pass("AA: High-risk assumption detection")
    else:
        results.add_fail("AA: High-risk assumption detection", 
                        "Expected at least 1 high-risk assumption")
    
    # Test 2: Overall risk calculation
    if aa_output.overall_assumption_risk in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]:
        results.add_pass("AA: Overall risk level")
    else:
        results.add_fail("AA: Overall risk level", 
                        f"Expected medium/high, got {aa_output.overall_assumption_risk}")


# =============================================================================
# AGENT 4: UNKNOWN DETECTION TESTS
# =============================================================================

def test_unknown_detection_agent(results: TestResults):
    """Test Unknown Detection Agent"""
    
    print("\n" + "="*80)
    print("TESTING UNKNOWN DETECTION AGENT")
    print("="*80)
    
    qp = QueryProcessor()
    fb = FactBoundaryAgent()
    aa = AssumptionAgent()
    ud = UnknownDetectionAgent()
    
    # Test 1: Detect missing critical fields
    query = "Am I eligible for schemes?"  # Missing everything
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    aa_output = aa.process(query, qp_output, fb_output)
    ud_output = ud.process(query, qp_output, fb_output, aa_output)
    
    if len(ud_output.missing_information) >= 3:
        results.add_pass("UD: Missing fields detection")
    else:
        results.add_fail("UD: Missing fields detection", 
                        f"Expected >=3 missing, got {len(ud_output.missing_information)}")
    
    # Test 2: Low completeness score for incomplete query
    if ud_output.information_completeness_score <= 0.4:
        results.add_pass("UD: Low completeness score")
    else:
        results.add_fail("UD: Low completeness score", 
                        f"Expected <=0.4, got {ud_output.information_completeness_score}")
    
    # Test 3: High completeness for complete query
    query = "I am 25 from Maharashtra, 3 LPA income, OBC, farmer, have Aadhaar"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    aa_output = aa.process(query, qp_output, fb_output)
    ud_output = ud.process(query, qp_output, fb_output, aa_output)
    
    if ud_output.information_completeness_score >= 0.6:
        results.add_pass("UD: High completeness score")
    else:
        results.add_fail("UD: High completeness score", 
                        f"Expected >=0.6, got {ud_output.information_completeness_score}")


# =============================================================================
# AGENT 5: TEMPORAL UNCERTAINTY TESTS
# =============================================================================

def test_temporal_agent(results: TestResults):
    """Test Temporal Uncertainty Agent"""
    
    print("\n" + "="*80)
    print("TESTING TEMPORAL UNCERTAINTY AGENT")
    print("="*80)
    
    qp = QueryProcessor()
    fb = FactBoundaryAgent()
    ud = UnknownDetectionAgent()
    aa = AssumptionAgent()
    tu = TemporalUncertaintyAgent()
    
    # Test 1: High time sensitivity for scheme queries
    query = "Am I eligible for current PM-KISAN scheme?"
    qp_output = qp.process(query)
    fb_output = fb.process(query, qp_output)
    aa_output = aa.process(query, qp_output, fb_output)
    ud_output = ud.process(query, qp_output, fb_output, aa_output)
    tu_output = tu.process(query, qp_output, fb_output, ud_output)
    
    if tu_output.time_sensitivity_level == RiskLevel.HIGH.value:
        results.add_pass("TU: High time sensitivity for schemes")
    else:
        results.add_fail("TU: High time sensitivity for schemes", 
                        f"Expected high, got {tu_output.time_sensitivity_level}")
    
    # Test 2: Recommended checks present
    if len(tu_output.recommended_fresh_checks) >= 2:
        results.add_pass("TU: Fresh check recommendations")
    else:
        results.add_fail("TU: Fresh check recommendations", 
                        f"Expected >=2 checks, got {len(tu_output.recommended_fresh_checks)}")


# =============================================================================
# AGENT 6: CONFIDENCE CALIBRATION TESTS
# =============================================================================

def test_confidence_calibration_agent(results: TestResults):
    """Test Confidence Calibration Agent"""
    
    print("\n" + "="*80)
    print("TESTING CONFIDENCE CALIBRATION AGENT")
    print("="*80)
    
    council = UncertaintyFirstAgentCouncil()
    
    # Test 1: Low confidence for vague query
    query = "Am I eligible?"
    full_results = council.process_query(query, verbose=False)
    cc_output = full_results['confidence_calibration']
    
    if cc_output.calibrated_confidence <= 30:
        results.add_pass("CC: Low confidence for vague query")
    else:
        results.add_fail("CC: Low confidence for vague query", 
                        f"Expected <=30, got {cc_output.calibrated_confidence}")
    
    # Test 2: Higher confidence for detailed query
    query = "I am 25 years old from Maharashtra with 3 LPA income, OBC category, farmer"
    full_results = council.process_query(query, verbose=False)
    cc_output = full_results['confidence_calibration']
    
    if cc_output.calibrated_confidence >= 40:
        results.add_pass("CC: Higher confidence for detailed query")
    else:
        results.add_fail("CC: Higher confidence for detailed query", 
                        f"Expected >=40, got {cc_output.calibrated_confidence}")
    
    # Test 3: Safety flag - unsafe for very vague
    query = "Schemes?"
    full_results = council.process_query(query, verbose=False)
    cc_output = full_results['confidence_calibration']
    
    if cc_output.safety_flag == SafetyFlag.UNSAFE_TO_ANSWER.value:
        results.add_pass("CC: Unsafe flag for very vague query")
    else:
        results.add_fail("CC: Unsafe flag for very vague query", 
                        f"Expected unsafe, got {cc_output.safety_flag}")


# =============================================================================
# AGENT 7: DECISION GUIDANCE TESTS
# =============================================================================

def test_decision_guidance_agent(results: TestResults):
    """Test Decision Guidance Agent"""
    
    print("\n" + "="*80)
    print("TESTING DECISION GUIDANCE AGENT")
    print("="*80)
    
    council = UncertaintyFirstAgentCouncil()
    
    # Test 1: No direct decision for unsafe queries
    query = "Eligible?"
    full_results = council.process_query(query, verbose=False)
    dg_output = full_results['decision_guidance']
    
    if dg_output.final_answer_style == "no_direct_decision":
        results.add_pass("DG: No direct decision for unsafe query")
    else:
        results.add_fail("DG: No direct decision for unsafe query", 
                        f"Expected no_direct_decision, got {dg_output.final_answer_style}")
    
    # Test 2: Recommended next steps present
    if len(dg_output.recommended_next_steps) >= 3:
        results.add_pass("DG: Next steps recommendations")
    else:
        results.add_fail("DG: Next steps recommendations", 
                        f"Expected >=3 steps, got {len(dg_output.recommended_next_steps)}")
    
    # Test 3: Explicit unknowns listed
    if len(dg_output.explicit_unknowns) >= 2:
        results.add_pass("DG: Explicit unknowns listed")
    else:
        results.add_fail("DG: Explicit unknowns listed", 
                        f"Expected >=2 unknowns, got {len(dg_output.explicit_unknowns)}")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_end_to_end_integration(results: TestResults):
    """Test end-to-end integration"""
    
    print("\n" + "="*80)
    print("TESTING END-TO-END INTEGRATION")
    print("="*80)
    
    council = UncertaintyFirstAgentCouncil()
    
    # Test 1: Full pipeline execution
    query = "I am 27, OBC, 4 LPA from Gujarat. PM Vishwakarma eligible?"
    
    try:
        full_results = council.process_query(query, verbose=False)
        results.add_pass("Integration: Full pipeline execution")
    except Exception as e:
        results.add_fail("Integration: Full pipeline execution", str(e))
    
    # Test 2: User response generation
    try:
        user_response = council.get_user_response(query)
        if len(user_response) > 100:  # Should be substantial
            results.add_pass("Integration: User response generation")
        else:
            results.add_fail("Integration: User response generation", 
                           "Response too short")
    except Exception as e:
        results.add_fail("Integration: User response generation", str(e))
    
    # Test 3: Multiple sequential queries
    try:
        queries = [
            "I am 25 from Maharashtra",
            "Income 3 LPA, OBC category",
            "Am I eligible?"
        ]
        for q in queries:
            council.get_user_response(q)
        results.add_pass("Integration: Multiple sequential queries")
    except Exception as e:
        results.add_fail("Integration: Multiple sequential queries", str(e))


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_edge_cases(results: TestResults):
    """Test edge cases and boundary conditions"""
    
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80)
    
    council = UncertaintyFirstAgentCouncil()
    
    # Test 1: Empty query
    try:
        response = council.get_user_response("")
        results.add_pass("Edge: Empty query handling")
    except Exception as e:
        results.add_fail("Edge: Empty query handling", str(e))
    
    # Test 2: Very long query
    long_query = "I am " + " ".join(["eligible"] * 100)
    try:
        response = council.get_user_response(long_query)
        results.add_pass("Edge: Very long query handling")
    except Exception as e:
        results.add_fail("Edge: Very long query handling", str(e))
    
    # Test 3: Special characters
    special_query = "Am I eligible??? $$$ ### @@@"
    try:
        response = council.get_user_response(special_query)
        results.add_pass("Edge: Special characters handling")
    except Exception as e:
        results.add_fail("Edge: Special characters handling", str(e))
    
    # Test 4: Non-English mixed query
    mixed_query = "I am 25 years old, मेरी income 3 LPA है"
    try:
        response = council.get_user_response(mixed_query)
        results.add_pass("Edge: Mixed language handling")
    except Exception as e:
        results.add_fail("Edge: Mixed language handling", str(e))


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run complete test suite"""
    
    print("\n" + "#"*80)
    print("#" + " "*30 + "TEST SUITE" + " "*38 + "#")
    print("#" + " "*15 + "Uncertainty-First Agent Council" + " "*32 + "#")
    print("#"*80)
    
    results = TestResults()
    
    # Run all test groups
    test_query_processor(results)
    test_fact_boundary_agent(results)
    test_assumption_agent(results)
    test_unknown_detection_agent(results)
    test_temporal_agent(results)
    test_confidence_calibration_agent(results)
    test_decision_guidance_agent(results)
    test_end_to_end_integration(results)
    test_edge_cases(results)
    
    # Print summary
    results.summary()
    
    return results


if __name__ == "__main__":
    run_all_tests()
