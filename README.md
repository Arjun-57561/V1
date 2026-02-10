# Uncertainty-First Agent Council

An Agentic AI System that Explicitly Models Unknowns for High-Stakes Indian Decision Support

## 📋 Overview

The Uncertainty-First Agent Council is a multi-agent system designed to prioritize **uncertainty awareness** over confident-sounding answers. Unlike traditional AI chatbots that optimize for fluent responses, this system explicitly identifies:

- ✅ What is **known** (facts)
- ❓ What is **unknown** (missing information)
- ⚠️ What is **assumed** (implicit assumptions)
- ⏰ What is **time-dependent** (temporal risks)

## 🎯 Problem Statement

Modern AI systems generate confident-looking answers even when:
- Information is incomplete
- Rules have changed or are outdated
- Critical details are ambiguous
- Assumptions are unverified

This is **especially dangerous** in high-stakes Indian contexts like:
- Government scheme eligibility
- Legal pre-screening
- Financial compliance

## 🏗️ System Architecture

The system consists of **7 specialized agents** working in sequence:

```
User Query
    ↓
[1] Query Processor → Cleans and extracts structured data
    ↓
[2] Fact Boundary Agent → Identifies definite facts
    ↓
[3] Assumption Agent → Makes assumptions explicit
    ↓
[4] Unknown Detection Agent → Flags missing information
    ↓
[5] Temporal Uncertainty Agent → Identifies time risks
    ↓
[6] Confidence Calibration Agent → Calculates calibrated confidence
    ↓
[7] Decision Guidance Agent → Generates transparent user guidance
    ↓
User-Friendly Response
```

## 🔧 Engineering Variables

### Input Variables
- Query complexity score
- Domain classification
- Information completeness

### Agent Parameters
- Confidence threshold (0-1)
- Uncertainty propagation factor
- Risk level weights

### Output Variables
- Calibrated confidence (0-100%)
- Known/Unknown classification
- Safety recommendation flag

## 📊 Agent Details

### Agent 1: Query Processor
**Purpose:** Clean and parse user query into structured fields

**Extracts:**
- Age, income, state/UT, caste category, occupation
- Documents mentioned
- Domain classification

**Output:** Structured JSON with entities and ambiguity flags

---

### Agent 2: Fact Boundary Agent
**Purpose:** Identify what is CERTAINLY TRUE

**Process:**
- Extracts only explicit facts from user query
- No inference or assumptions
- Calculates clarity score (0-1)

**Output:** List of known facts with sources

---

### Agent 3: Assumption Agent
**Purpose:** Make implicit assumptions EXPLICIT

**Identifies:**
- Reasonable assumptions (e.g., Indian citizenship)
- Risky assumptions (e.g., documented income)
- Impact if assumptions are wrong

**Output:** List of assumptions with risk levels (low/medium/high)

---

### Agent 4: Unknown Detection Agent
**Purpose:** Flag MISSING critical information

**Checks:**
- Required fields per domain
- Importance level of each missing field
- Consequences if ignored

**Output:** List of unknowns with importance and consequences

---

### Agent 5: Temporal Uncertainty Agent
**Purpose:** Identify TIME-DEPENDENT risks

**Analyzes:**
- Whether rules/laws may have changed
- Scheme deadlines and cycles
- Recommended verification checks

**Output:** Time sensitivity level + recommended fresh checks

---

### Agent 6: Confidence Calibration Agent
**Purpose:** Convert uncertainty into CALIBRATED CONFIDENCE SCORE

**Algorithm:**
```python
base_confidence = 80

# Deductions
missing_info_penalty = up to 30 points
assumption_risk_penalty = up to 20 points
time_sensitivity_penalty = up to 20 points

# Bonuses
clarity_bonus = up to 10 points

calibrated_confidence = clamp(base - penalties + bonuses, 0, 100)
```

**Output:** Confidence score (0-100) + safety flag

---

### Agent 7: Decision Guidance Agent
**Purpose:** Generate ACTIONABLE user guidance

**Produces:**
- User-friendly summary
- Explicit knowns, unknowns, assumptions
- Recommended next steps
- Final answer style (no decision / cautious / direct)

**Output:** Transparent, actionable response for user

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
No external dependencies required (uses only Python standard library)
```

### Installation
```bash
# Clone or download the file
# No pip install needed - pure Python!
```

### Basic Usage

```python
from uncertainty_agent_council import UncertaintyFirstAgentCouncil

# Initialize the council
council = UncertaintyFirstAgentCouncil()

# Process a user query
user_query = "I am 25 years old from Maharashtra with 3 LPA income. Am I eligible for PM-KISAN?"

# Get the final user response
response = council.get_user_response(user_query)
print(response)
```

### Advanced Usage (Verbose Mode)

```python
# Get detailed outputs from all agents
results = council.process_query(user_query, verbose=True)

# Access individual agent outputs
query_processor_output = results["query_processor"]
fact_boundary_output = results["fact_boundary"]
assumption_output = results["assumption_agent"]
unknown_output = results["unknown_detection"]
temporal_output = results["temporal_uncertainty"]
confidence_output = results["confidence_calibration"]
guidance_output = results["decision_guidance"]

# Get JSON outputs
from uncertainty_agent_council import QueryProcessor
qp = QueryProcessor()
json_output = qp.to_json(query_processor_output)
```

## 📖 Example Scenarios

### Example 1: High Clarity Query

**Input:**
```
"I am 25 years old from Maharashtra with annual income of 3 LPA 
and belong to OBC category. Am I eligible for PM-KISAN scheme?"
```

**System Analysis:**
- ✅ Known: Age (25), State (Maharashtra), Income (3 LPA), Category (OBC)
- ❓ Unknown: Whether user owns agricultural land, documented income proof
- ⚠️ Assumptions: Indian citizenship, OBC certificate available
- 🎯 Confidence: ~65% (moderate)
- 🚦 Safety: Answer with caution

**Output:**
```
⚡ Based on the limited information provided, I can offer a tentative assessment...

📊 Confidence Level: 65%
🚦 Safety Assessment: Answer With Caution

✅ WHAT WE KNOW:
   1. User's age is 25 years
   2. Annual income is approximately ₹300,000.00
   3. User is from Maharashtra
   4. User belongs to OBC category

❓ WHAT WE DON'T KNOW:
   1. Whether you own agricultural land (PM-KISAN requires it)
   2. Whether you have documented income proof
   
⚠️ ASSUMPTIONS WE'RE MAKING:
   1. Assuming user is an Indian citizen
   2. Assuming user has valid OBC category certificate
   
📋 RECOMMENDED NEXT STEPS:
   1. Verify if you own agricultural land
   2. Obtain official income certificate
   3. Check latest PM-KISAN eligibility on official portal
```

---

### Example 2: Low Clarity Query

**Input:**
```
"Am I eligible for any schemes?"
```

**System Analysis:**
- ✅ Known: Nothing specific
- ❓ Unknown: Age, income, state, category, specific scheme
- 🎯 Confidence: ~15% (very low)
- 🚦 Safety: Unsafe to answer

**Output:**
```
⚠️ I cannot provide a reliable answer due to insufficient information 
(confidence: 15%). Too many critical details are missing.

❓ WHAT WE DON'T KNOW:
   1. Age: User's age is not specified
   2. Annual Income: Income information missing
   3. State/UT: Location not mentioned
   4. Caste Category: Not specified
   
📋 RECOMMENDED NEXT STEPS:
   1. Provide your current age
   2. Specify your annual/monthly income
   3. Mention your state of residence
   4. Clarify your caste category
```

## 🧪 Testing

### Run Demo
```bash
python uncertainty_agent_council.py
```

This will run 4 test queries demonstrating different clarity levels.

### Custom Testing
```python
council = UncertaintyFirstAgentCouncil()

test_queries = [
    "Your test query 1",
    "Your test query 2",
]

for query in test_queries:
    response = council.get_user_response(query)
    print(response)
```

## 📈 Confidence Calibration Logic

```
Base Confidence: 80

Penalties:
- Missing critical info: -5 to -30 points (based on importance)
- High assumption risk: -20 points
- Medium assumption risk: -12 points
- High time sensitivity: -20 points
- Medium time sensitivity: -10 points

Bonuses:
- High clarity score (≥0.8): +10 points

Final = clamp(base - penalties + bonuses, 0, 100)

Safety Flags:
- confidence < 30 OR 3+ critical missing → UNSAFE
- confidence < 60 OR 1-2 critical missing → CAUTION
- Otherwise → SAFE
```

## 🎨 Customization

### Adding New Domains

```python
# In QueryProcessor class
SCHEME_KEYWORDS = ['your', 'keywords']

# In UnknownDetectionAgent class
CRITICAL_FIELDS = {
    "your_new_domain": ["field1", "field2", "field3"]
}
```

### Adjusting Confidence Penalties

```python
# In ConfidenceCalibrationAgent class

def _calculate_missing_info_penalty(self, unknown_output):
    # Modify penalty logic here
    base_penalty = int((1.0 - completeness) * 30)  # Change multiplier
    return base_penalty
```

## 📊 Output JSON Schema

Each agent returns structured JSON. Example from Decision Guidance Agent:

```json
{
  "final_answer_style": "cautious_tentative_decision",
  "user_friendly_summary": "⚡ Based on limited information...",
  "explicit_knowns": [
    "User's age is 25 years",
    "Annual income is ₹300,000"
  ],
  "explicit_unknowns": [
    "Whether you have income certificate"
  ],
  "assumptions_highlighted": [
    "Assuming user is Indian citizen"
  ],
  "calibrated_confidence": 65,
  "safety_flag": "answer_with_caution",
  "recommended_next_steps": [
    "Obtain income certificate",
    "Check official scheme portal"
  ]
}
```

## 🔐 Safety Features

1. **Never guesses** - If info is missing, it says so explicitly
2. **No hallucinations** - Confidence drops when uncertainty is high
3. **Time-aware** - Warns when rules may have changed
4. **Assumption transparency** - Shows what it's assuming
5. **Safety flags** - Prevents misleading answers

## 🎓 Educational Value

This project demonstrates:
- Multi-agent systems architecture
- Uncertainty quantification in AI
- Epistemic vs aleatoric uncertainty
- Decision support system design
- Prompt engineering at scale
- AI safety principles

## 🚧 Limitations

1. **No LLM integration** - This is a rule-based parser, not using GPT/Claude API
2. **No real-time data** - Doesn't fetch current scheme details
3. **Limited entity extraction** - Uses regex, not NER models
4. **English-only** - Doesn't handle Hindi/regional languages
5. **No learning** - Doesn't improve from user feedback

## 🔮 Future Enhancements

1. **LLM Integration** - Use GPT-4/Claude for each agent
2. **RAG System** - Retrieve actual scheme documents
3. **Multilingual Support** - Hindi, Marathi, Tamil, etc.
4. **User Feedback Loop** - Learn from corrections
5. **Web Interface** - Streamlit/Gradio UI
6. **Database** - Store common queries and responses

## 👥 Team

- **Amitkumar Racha** - 24070126501
- **Bontha Mallikarjun Reddy** - 23070126026
- **Neil Cardoz** - 23070126079

**Mentors:**
- Dr. Aniket Shahade
- Dr. Sumit Kumar

## 📄 License

Educational project for PBL coursework.

## 🙏 Acknowledgments

Inspired by AI safety research on uncertainty quantification and epistemic awareness in language models.

---

**Remember:** This system prioritizes **honesty over fluency**. It would rather say "I don't know" than give a confidently wrong answer.
