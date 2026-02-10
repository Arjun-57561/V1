"""
QUICK START GUIDE
Uncertainty-First Agent Council

Get started in 5 minutes!
"""

# =============================================================================
# STEP 1: Import the Council
# =============================================================================

from uncertainty_agent_council import UncertaintyFirstAgentCouncil

# =============================================================================
# STEP 2: Create Council Instance
# =============================================================================

council = UncertaintyFirstAgentCouncil()

# =============================================================================
# STEP 3: Process Your First Query
# =============================================================================

# Example: Government scheme eligibility
query = """
I am 25 years old from Maharashtra with annual income of 3 LPA.
I belong to OBC category. Am I eligible for PM-KISAN scheme?
"""

# Get user-friendly response
response = council.get_user_response(query)
print(response)

# =============================================================================
# STEP 4: Try Different Query Types
# =============================================================================

# Clear query (high confidence)
clear_query = """
I am 23 years old from Karnataka, earning 2.5 LPA annually.
I belong to SC category and work as a farmer. I have Aadhaar and ration card.
Am I eligible for PM-KISAN Samman Nidhi Yojana?
"""

# Vague query (low confidence)
vague_query = "Am I eligible for any schemes?"

# Time-sensitive query
time_query = "What are the current tax benefits for senior citizens in 2025?"

# Process any of them
response = council.get_user_response(clear_query)
print(response)

# =============================================================================
# STEP 5: Access Detailed Results (Advanced)
# =============================================================================

# Get full agent outputs
results = council.process_query(query, verbose=False)

# Access specific agent outputs
confidence = results['confidence_calibration'].calibrated_confidence
safety_flag = results['confidence_calibration'].safety_flag
knowns = results['decision_guidance'].explicit_knowns
unknowns = results['decision_guidance'].explicit_unknowns

print(f"\nConfidence: {confidence}%")
print(f"Safety: {safety_flag}")
print(f"Known facts: {len(knowns)}")
print(f"Missing info: {len(unknowns)}")

# =============================================================================
# STEP 6: Batch Processing (Optional)
# =============================================================================

queries = [
    "I am 30 years old, OBC, 5 LPA income. Any schemes?",
    "Senior citizen, pension income 4 lakhs. Tax benefits?",
    "22 years, student, need scholarship. Eligible?",
]

for q in queries:
    result = council.process_query(q, verbose=False)
    conf = result['confidence_calibration'].calibrated_confidence
    print(f"\nQuery: {q[:50]}...")
    print(f"Confidence: {conf}%")

# =============================================================================
# THAT'S IT! You're ready to use the Uncertainty-First Agent Council
# =============================================================================

"""
KEY FEATURES TO REMEMBER:

1. ✅ Explicitly shows what is KNOWN
2. ❓ Explicitly shows what is UNKNOWN
3. ⚠️ Makes ASSUMPTIONS transparent
4. 📊 Provides CALIBRATED confidence (not overconfident)
5. 🚦 Includes SAFETY FLAGS (safe/caution/unsafe to answer)
6. 📋 Recommends NEXT STEPS

The system prioritizes HONESTY over FLUENCY.
It would rather say "I don't know" than give a wrong confident answer.
"""

# =============================================================================
# COMMON USE CASES
# =============================================================================

def use_case_government_schemes():
    """Check government scheme eligibility"""
    council = UncertaintyFirstAgentCouncil()
    
    query = "I am 27, OBC, 4 LPA, from Gujarat. PM Vishwakarma Yojana eligible?"
    response = council.get_user_response(query)
    print(response)


def use_case_legal_screening():
    """Pre-screen legal queries"""
    council = UncertaintyFirstAgentCouncil()
    
    query = """
    I want to file consumer complaint against mobile company.
    Defective phone purchased 3 months ago in Mumbai.
    """
    response = council.get_user_response(query)
    print(response)


def use_case_financial_compliance():
    """Check tax/financial compliance"""
    council = UncertaintyFirstAgentCouncil()
    
    query = "My annual income is 12 lakhs. Do I need to file ITR?"
    response = council.get_user_response(query)
    print(response)


# =============================================================================
# UNDERSTANDING OUTPUT
# =============================================================================

"""
SAMPLE OUTPUT STRUCTURE:

================================================================================
RESPONSE TO YOUR QUERY
================================================================================

⚡ Based on the limited information provided, I can offer a tentative assessment
for your government scheme query (confidence: 65%). However, several assumptions
have been made and important details are missing.

📊 Confidence Level: 65%
🚦 Safety Assessment: Answer With Caution

✅ WHAT WE KNOW:
   1. User's age is 25 years
   2. Annual income is approximately ₹300,000.00
   3. User is from Maharashtra
   4. User belongs to OBC category

❓ WHAT WE DON'T KNOW:
   1. Required Documents: No information about available documents
   2. Specific Scheme Name: User hasn't specified which scheme

⚠️ ASSUMPTIONS WE'RE MAKING:
   1. Assuming user is an Indian citizen
   2. Assuming reported income is documented and verifiable
   3. Assuming user has valid OBC category certificate

📋 RECOMMENDED NEXT STEPS:
   1. Gather required documents (Aadhaar, domicile certificate, etc.)
   2. Check official scheme portal for latest eligibility criteria
   3. Verify scheme is still active and accepting applications

================================================================================
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Q: Why is confidence so low for my query?
A: The system detects missing critical information. Add more details like:
   - Age, income, state, category
   - Specific scheme name
   - Documents you have

Q: Why does it say "unsafe to answer"?
A: Too many unknowns or high-risk assumptions. The system won't guess.
   This is a FEATURE, not a bug - it prevents misleading answers.

Q: How do I increase confidence?
A: Provide more specific information in your query:
   - Instead of "around 30 years", say "30 years old"
   - Instead of "some income", say "3 LPA annual income"
   - Mention your state, category, specific scheme name

Q: Can it fetch real-time scheme data?
A: No, this is a rule-based parser. For production, integrate with:
   - LLM API (GPT-4, Claude)
   - RAG system for scheme documents
   - Government APIs for real-time data
"""

# =============================================================================
# NEXT STEPS FOR YOUR PROJECT
# =============================================================================

"""
TO ENHANCE THIS SYSTEM:

1. LLM Integration
   - Replace rule-based parsing with GPT-4/Claude API calls
   - Each agent becomes an LLM with specialized prompt

2. RAG System
   - Add vector database for scheme documents
   - Retrieve actual eligibility criteria

3. Real-time Data
   - Fetch from government portals
   - Verify scheme status, deadlines

4. UI Development
   - Streamlit/Gradio web interface
   - Mobile app

5. Feedback Loop
   - Collect user corrections
   - Improve entity extraction

See README.md for detailed implementation guide.
"""
