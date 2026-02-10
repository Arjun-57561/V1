"""
Uncertainty-First Agent Council
A Multi-Agent System for Explicit Uncertainty Modeling in Indian Decision Support

Author: PBL Team
Date: 2025
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


# ==================== ENUMS & DATA STRUCTURES ====================

class Domain(str, Enum):
    GOVERNMENT_SCHEME = "government_scheme"
    LEGAL_PRESCREENING = "legal_prescreening"
    FINANCIAL_COMPLIANCE = "financial_compliance"
    OTHER = "other"


class Category(str, Enum):
    SC = "SC"
    ST = "ST"
    OBC = "OBC"
    GENERAL = "General"
    EWS = "EWS"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImportanceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyFlag(str, Enum):
    SAFE_TO_ANSWER = "safe_to_answer"
    ANSWER_WITH_CAUTION = "answer_with_caution"
    UNSAFE_TO_ANSWER = "unsafe_to_answer"


class AnswerStyle(str, Enum):
    NO_DIRECT_DECISION = "no_direct_decision"
    CAUTIOUS_TENTATIVE_DECISION = "cautious_tentative_decision"
    DIRECT_DECISION = "direct_decision"


# ==================== AGENT 1: QUERY PROCESSOR ====================

@dataclass
class QueryEntities:
    age: Optional[int] = None
    monthly_income: Optional[float] = None
    annual_income: Optional[float] = None
    state_or_ut: Optional[str] = None
    category: Optional[str] = None
    occupation: Optional[str] = None
    documents_mentioned: List[str] = None
    
    def __post_init__(self):
        if self.documents_mentioned is None:
            self.documents_mentioned = []


@dataclass
class QueryProcessorOutput:
    cleaned_query: str
    detected_domain: str
    entities: QueryEntities
    ambiguity_flags: List[str]
    notes: str


class QueryProcessor:
    """
    Agent 1: Query Processor / Pre-Parsing Agent
    Cleans user's natural-language question and extracts structured fields
    """
    
    SCHEME_KEYWORDS = ['scheme', 'yojana', 'subsidy', 'grant', 'benefit', 'eligibility']
    LEGAL_KEYWORDS = ['legal', 'law', 'court', 'case', 'advocate', 'rights']
    FINANCIAL_KEYWORDS = ['tax', 'loan', 'investment', 'finance', 'banking', 'gst']
    
    STATES_UT = [
        'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
        'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
        'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
        'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
        'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
        'delhi', 'jammu and kashmir', 'ladakh', 'puducherry', 'chandigarh',
        'andaman and nicobar', 'dadra and nagar haveli', 'daman and diu', 'lakshadweep'
    ]
    
    DOCUMENTS = [
        'aadhar', 'aadhaar', 'pan', 'income certificate', 'caste certificate',
        'domicile certificate', 'ration card', 'voter id', 'driving license',
        'passport', 'birth certificate', 'bank statement', 'salary slip'
    ]
    
    def process(self, user_query: str) -> QueryProcessorOutput:
        """Main processing function"""
        
        # Clean query
        cleaned = self._clean_query(user_query)
        
        # Detect domain
        domain = self._detect_domain(cleaned)
        
        # Extract entities
        entities = self._extract_entities(cleaned)
        
        # Identify ambiguities
        ambiguity_flags = self._identify_ambiguities(cleaned, entities)
        
        # Generate notes
        notes = self._generate_notes(entities, ambiguity_flags)
        
        return QueryProcessorOutput(
            cleaned_query=cleaned,
            detected_domain=domain,
            entities=entities,
            ambiguity_flags=ambiguity_flags,
            notes=notes
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        # Convert to lowercase for processing (we'll keep original case in cleaned_query)
        return cleaned.strip()
    
    def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query"""
        query_lower = query.lower()
        
        scheme_score = sum(1 for kw in self.SCHEME_KEYWORDS if kw in query_lower)
        legal_score = sum(1 for kw in self.LEGAL_KEYWORDS if kw in query_lower)
        financial_score = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in query_lower)
        
        max_score = max(scheme_score, legal_score, financial_score)
        
        if max_score == 0:
            return Domain.OTHER.value
        elif scheme_score == max_score:
            return Domain.GOVERNMENT_SCHEME.value
        elif legal_score == max_score:
            return Domain.LEGAL_PRESCREENING.value
        else:
            return Domain.FINANCIAL_COMPLIANCE.value
    
    def _extract_entities(self, query: str) -> QueryEntities:
        """Extract structured entities from query"""
        query_lower = query.lower()
        entities = QueryEntities()
        
        # Extract age
        age_patterns = [
            r'\b(\d{1,2})\s*(?:years?|yrs?)\s*old\b',
            r'\bage\s*(?:is\s*)?(\d{1,2})\b',
            r'\b(\d{1,2})\s*(?:year|yr)\b'
        ]
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities.age = int(match.group(1))
                break
        
        # Extract income
        income_patterns = [
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lpa|lakhs?\s*per\s*annum)',
            r'annual\s*income\s*(?:of\s*)?(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'monthly\s*income\s*(?:of\s*)?(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'(?:earn|earning|salary)\s*(?:of\s*)?(?:rs\.?\s*)?(\d+(?:,\d+)*)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                income_str = match.group(1).replace(',', '')
                income_val = float(income_str)
                
                if 'lpa' in match.group(0) or 'lakhs per annum' in match.group(0):
                    entities.annual_income = income_val * 100000
                elif 'annual' in match.group(0):
                    entities.annual_income = income_val
                elif 'monthly' in match.group(0):
                    entities.monthly_income = income_val
                    entities.annual_income = income_val * 12
                break
        
        # Extract state
        for state in self.STATES_UT:
            if state in query_lower:
                entities.state_or_ut = state.title()
                break
        
        # Extract category
        for cat in Category:
            if cat.value.lower() in query_lower:
                entities.category = cat.value
                break
        
        # Extract occupation
        occupation_patterns = [
            r'(?:work\s*as\s*a?\s*)(\w+)',
            r'(?:occupation\s*(?:is\s*)?)(\w+)',
            r'(?:job\s*(?:is\s*)?)(\w+)',
            r'\b(farmer|student|teacher|engineer|doctor|labor|labourer)\b'
        ]
        for pattern in occupation_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities.occupation = match.group(1).capitalize()
                break
        
        # Extract documents
        for doc in self.DOCUMENTS:
            if doc in query_lower:
                entities.documents_mentioned.append(doc.title())
        
        return entities
    
    def _identify_ambiguities(self, query: str, entities: QueryEntities) -> List[str]:
        """Identify ambiguities in the query"""
        flags = []
        
        if entities.age is None:
            flags.append("Age not specified")
        
        if entities.annual_income is None and entities.monthly_income is None:
            flags.append("Income information missing")
        
        if entities.state_or_ut is None:
            flags.append("State/UT not mentioned (may affect scheme eligibility)")
        
        if entities.category is None:
            flags.append("Caste category not specified")
        
        # Check for vague terms
        vague_terms = ['maybe', 'approximately', 'around', 'roughly', 'about', 'I think']
        if any(term in query.lower() for term in vague_terms):
            flags.append("Query contains uncertainty markers (maybe, approximately, etc.)")
        
        return flags
    
    def _generate_notes(self, entities: QueryEntities, ambiguity_flags: List[str]) -> str:
        """Generate explanatory notes"""
        extracted_count = sum([
            entities.age is not None,
            entities.annual_income is not None,
            entities.state_or_ut is not None,
            entities.category is not None,
            entities.occupation is not None,
            len(entities.documents_mentioned) > 0
        ])
        
        if extracted_count >= 4:
            return f"Successfully extracted {extracted_count}/6 key entities. Query is relatively clear."
        elif extracted_count >= 2:
            return f"Extracted {extracted_count}/6 key entities. Several critical fields missing. {len(ambiguity_flags)} ambiguities detected."
        else:
            return f"Only extracted {extracted_count}/6 key entities. Query is highly ambiguous with insufficient information."
    
    def to_json(self, output: QueryProcessorOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "cleaned_query": output.cleaned_query,
            "detected_domain": output.detected_domain,
            "entities": {
                "age": output.entities.age,
                "monthly_income": output.entities.monthly_income,
                "annual_income": output.entities.annual_income,
                "state_or_ut": output.entities.state_or_ut,
                "category": output.entities.category,
                "occupation": output.entities.occupation,
                "documents_mentioned": output.entities.documents_mentioned
            },
            "ambiguity_flags": output.ambiguity_flags,
            "notes": output.notes
        }, indent=2)


# ==================== AGENT 2: FACT BOUNDARY ====================

@dataclass
class KnownFact:
    fact_id: str
    description: str
    source: str  # "user_query" | "parsed_entity"


@dataclass
class FactBoundaryOutput:
    known_facts: List[KnownFact]
    clarity_score: float  # 0.0 to 1.0
    explanatory_notes: str


class FactBoundaryAgent:
    """
    Agent 2: Fact Boundary Agent
    Identifies what is CERTAINLY TRUE given user query and parsed entities
    """
    
    def process(self, user_query: str, query_processor_output: QueryProcessorOutput) -> FactBoundaryOutput:
        """Identify definite facts"""
        
        known_facts = []
        fact_counter = 1
        
        entities = query_processor_output.entities
        
        # Extract facts from entities
        if entities.age is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"User's age is {entities.age} years",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.annual_income is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"Annual income is approximately ₹{entities.annual_income:,.2f}",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.monthly_income is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"Monthly income is approximately ₹{entities.monthly_income:,.2f}",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.state_or_ut is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"User is from {entities.state_or_ut}",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.category is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"User belongs to {entities.category} category",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.occupation is not None:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"User's occupation is {entities.occupation}",
                source="parsed_entity"
            ))
            fact_counter += 1
        
        if entities.documents_mentioned:
            known_facts.append(KnownFact(
                fact_id=f"F{fact_counter}",
                description=f"User mentioned having: {', '.join(entities.documents_mentioned)}",
                source="user_query"
            ))
            fact_counter += 1
        
        # Calculate clarity score
        clarity_score = self._calculate_clarity_score(query_processor_output, len(known_facts))
        
        # Generate explanatory notes
        explanatory_notes = self._generate_notes(known_facts, clarity_score, query_processor_output)
        
        return FactBoundaryOutput(
            known_facts=known_facts,
            clarity_score=clarity_score,
            explanatory_notes=explanatory_notes
        )
    
    def _calculate_clarity_score(self, qp_output: QueryProcessorOutput, fact_count: int) -> float:
        """Calculate how clear the user's situation is described (0-1)"""
        
        # Base score from number of facts
        base_score = min(fact_count / 6.0, 1.0)  # 6 key entities max
        
        # Penalty for ambiguity flags
        ambiguity_penalty = len(qp_output.ambiguity_flags) * 0.1
        
        # Penalty for vague query
        if len(qp_output.cleaned_query.split()) < 10:
            base_score -= 0.1
        
        clarity_score = max(0.0, min(1.0, base_score - ambiguity_penalty))
        
        return round(clarity_score, 2)
    
    def _generate_notes(self, facts: List[KnownFact], clarity_score: float, 
                       qp_output: QueryProcessorOutput) -> str:
        """Generate explanatory notes"""
        
        if clarity_score >= 0.7:
            return f"High clarity: {len(facts)} definite facts extracted. User's situation is well-described."
        elif clarity_score >= 0.4:
            return f"Medium clarity: {len(facts)} facts known, but {len(qp_output.ambiguity_flags)} ambiguities present."
        else:
            return f"Low clarity: Only {len(facts)} facts confirmed. Significant information gaps make reliable assessment difficult."
    
    def to_json(self, output: FactBoundaryOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "known_facts": [
                {
                    "fact_id": fact.fact_id,
                    "description": fact.description,
                    "source": fact.source
                }
                for fact in output.known_facts
            ],
            "clarity_score": output.clarity_score,
            "explanatory_notes": output.explanatory_notes
        }, indent=2)


# ==================== AGENT 3: ASSUMPTION AGENT ====================

@dataclass
class Assumption:
    assumption_id: str
    description: str
    risk_level: str  # RiskLevel
    impact_if_wrong: str


@dataclass
class AssumptionAgentOutput:
    assumptions: List[Assumption]
    overall_assumption_risk: str  # RiskLevel
    notes: str


class AssumptionAgent:
    """
    Agent 3: Assumption Agent
    Makes implicit assumptions explicit and assesses their risk
    """
    
    def process(self, user_query: str, query_processor_output: QueryProcessorOutput,
                fact_boundary_output: FactBoundaryOutput) -> AssumptionAgentOutput:
        """Identify and assess assumptions"""
        
        assumptions = []
        assumption_counter = 1
        
        entities = query_processor_output.entities
        domain = query_processor_output.detected_domain
        
        # Assumption: Indian citizenship
        if domain == Domain.GOVERNMENT_SCHEME.value:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming user is an Indian citizen",
                risk_level=RiskLevel.MEDIUM.value,
                impact_if_wrong="Non-citizens are typically ineligible for most government schemes"
            ))
            assumption_counter += 1
        
        # Assumption: Income is legally reported
        if entities.annual_income is not None or entities.monthly_income is not None:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming reported income is documented and verifiable",
                risk_level=RiskLevel.HIGH.value,
                impact_if_wrong="Schemes require official income certificates; undocumented income won't qualify"
            ))
            assumption_counter += 1
        
        # Assumption: Category certificate exists
        if entities.category is not None and entities.category != Category.GENERAL.value:
            if "certificate" not in user_query.lower():
                assumptions.append(Assumption(
                    assumption_id=f"A{assumption_counter}",
                    description=f"Assuming user has valid {entities.category} category certificate",
                    risk_level=RiskLevel.HIGH.value,
                    impact_if_wrong="Category-based benefits require official caste certificates from competent authority"
                ))
                assumption_counter += 1
        
        # Assumption: Current scheme rules apply
        if domain == Domain.GOVERNMENT_SCHEME.value:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming current scheme eligibility rules haven't changed recently",
                risk_level=RiskLevel.MEDIUM.value,
                impact_if_wrong="Schemes frequently update criteria, income limits, or close applications"
            ))
            assumption_counter += 1
        
        # Assumption: Age is current
        if entities.age is not None:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming stated age is current (not past or future age)",
                risk_level=RiskLevel.LOW.value,
                impact_if_wrong="Some schemes have strict age cutoffs; timing matters"
            ))
            assumption_counter += 1
        
        # Assumption: No missing dependents
        if domain == Domain.GOVERNMENT_SCHEME.value:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming household size/dependents not relevant or already considered",
                risk_level=RiskLevel.MEDIUM.value,
                impact_if_wrong="Many schemes consider family size, dependents, or household income"
            ))
            assumption_counter += 1
        
        # Assumption: Documents are obtainable
        if not entities.documents_mentioned and domain == Domain.GOVERNMENT_SCHEME.value:
            assumptions.append(Assumption(
                assumption_id=f"A{assumption_counter}",
                description="Assuming user can obtain required documents if needed",
                risk_level=RiskLevel.HIGH.value,
                impact_if_wrong="Document procurement can be complex, time-consuming, or require prerequisites"
            ))
            assumption_counter += 1
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(assumptions)
        
        # Generate notes
        notes = self._generate_notes(assumptions, overall_risk)
        
        return AssumptionAgentOutput(
            assumptions=assumptions,
            overall_assumption_risk=overall_risk,
            notes=notes
        )
    
    def _calculate_overall_risk(self, assumptions: List[Assumption]) -> str:
        """Calculate overall assumption risk level"""
        
        if not assumptions:
            return RiskLevel.LOW.value
        
        risk_counts = {
            RiskLevel.HIGH.value: sum(1 for a in assumptions if a.risk_level == RiskLevel.HIGH.value),
            RiskLevel.MEDIUM.value: sum(1 for a in assumptions if a.risk_level == RiskLevel.MEDIUM.value),
            RiskLevel.LOW.value: sum(1 for a in assumptions if a.risk_level == RiskLevel.LOW.value)
        }
        
        # If any high-risk assumptions, overall is high
        if risk_counts[RiskLevel.HIGH.value] >= 2:
            return RiskLevel.HIGH.value
        elif risk_counts[RiskLevel.HIGH.value] >= 1 or risk_counts[RiskLevel.MEDIUM.value] >= 3:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value
    
    def _generate_notes(self, assumptions: List[Assumption], overall_risk: str) -> str:
        """Generate explanatory notes"""
        
        high_count = sum(1 for a in assumptions if a.risk_level == RiskLevel.HIGH.value)
        
        if overall_risk == RiskLevel.HIGH.value:
            return f"HIGH RISK: {high_count} critical assumptions made. If any are wrong, the answer will be misleading."
        elif overall_risk == RiskLevel.MEDIUM.value:
            return f"MEDIUM RISK: {len(assumptions)} assumptions identified. Verification recommended before proceeding."
        else:
            return f"LOW RISK: {len(assumptions)} low-risk assumptions. Mostly standard context for this query type."
    
    def to_json(self, output: AssumptionAgentOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "assumptions": [
                {
                    "assumption_id": a.assumption_id,
                    "description": a.description,
                    "risk_level": a.risk_level,
                    "impact_if_wrong": a.impact_if_wrong
                }
                for a in output.assumptions
            ],
            "overall_assumption_risk": output.overall_assumption_risk,
            "notes": output.notes
        }, indent=2)


# ==================== AGENT 4: UNKNOWN DETECTION ====================

@dataclass
class MissingInformation:
    unknown_id: str
    field_name: str
    description: str
    importance_level: str  # ImportanceLevel
    consequence_if_ignored: str


@dataclass
class UnknownDetectionOutput:
    missing_information: List[MissingInformation]
    information_completeness_score: float  # 0.0 to 1.0
    notes: str


class UnknownDetectionAgent:
    """
    Agent 4: Unknown Detection Agent
    Identifies missing but critical information required for safe recommendations
    """
    
    # Critical fields per domain
    CRITICAL_FIELDS = {
        Domain.GOVERNMENT_SCHEME.value: [
            "age", "annual_income", "state_or_ut", "category", "documents"
        ],
        Domain.LEGAL_PRESCREENING.value: [
            "state_or_ut", "case_details", "timeline"
        ],
        Domain.FINANCIAL_COMPLIANCE.value: [
            "annual_income", "financial_year", "documentation"
        ]
    }
    
    def process(self, user_query: str, query_processor_output: QueryProcessorOutput,
                fact_boundary_output: FactBoundaryOutput,
                assumption_agent_output: AssumptionAgentOutput) -> UnknownDetectionOutput:
        """Identify missing critical information"""
        
        missing_info = []
        unknown_counter = 1
        
        entities = query_processor_output.entities
        domain = query_processor_output.detected_domain
        
        # Check for missing critical fields based on domain
        if domain == Domain.GOVERNMENT_SCHEME.value:
            
            # Age missing
            if entities.age is None:
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="Age",
                    description="User's age is not specified",
                    importance_level=ImportanceLevel.HIGH.value,
                    consequence_if_ignored="Most schemes have strict age limits (e.g., 18-40, below 60, etc.)"
                ))
                unknown_counter += 1
            
            # Income missing
            if entities.annual_income is None:
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="Annual Income",
                    description="User's annual income is not specified",
                    importance_level=ImportanceLevel.HIGH.value,
                    consequence_if_ignored="Income limits are primary eligibility criteria for most welfare schemes"
                ))
                unknown_counter += 1
            
            # State missing
            if entities.state_or_ut is None:
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="State/UT",
                    description="User's state or union territory is unknown",
                    importance_level=ImportanceLevel.HIGH.value,
                    consequence_if_ignored="Many schemes are state-specific with different rules and availability"
                ))
                unknown_counter += 1
            
            # Category missing
            if entities.category is None:
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="Caste Category",
                    description="User's caste category (SC/ST/OBC/General/EWS) is not specified",
                    importance_level=ImportanceLevel.MEDIUM.value,
                    consequence_if_ignored="Reserved category benefits and quotas depend on this information"
                ))
                unknown_counter += 1
            
            # Documents missing
            if not entities.documents_mentioned:
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="Required Documents",
                    description="No information about available documents",
                    importance_level=ImportanceLevel.HIGH.value,
                    consequence_if_ignored="Schemes require specific documents (income cert, domicile, category cert, etc.)"
                ))
                unknown_counter += 1
            
            # Scheme name not mentioned
            if not any(word in user_query.lower() for word in ['pm', 'pradhan mantri', 'yojana', 'scheme name']):
                missing_info.append(MissingInformation(
                    unknown_id=f"U{unknown_counter}",
                    field_name="Specific Scheme Name",
                    description="User hasn't specified which scheme they're asking about",
                    importance_level=ImportanceLevel.MEDIUM.value,
                    consequence_if_ignored="Different schemes have vastly different criteria and benefits"
                ))
                unknown_counter += 1
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            domain, entities, len(missing_info)
        )
        
        # Generate notes
        notes = self._generate_notes(missing_info, completeness_score)
        
        return UnknownDetectionOutput(
            missing_information=missing_info,
            information_completeness_score=completeness_score,
            notes=notes
        )
    
    def _calculate_completeness_score(self, domain: str, entities: QueryEntities, 
                                     missing_count: int) -> float:
        """Calculate information completeness (0-1)"""
        
        # Count how many critical fields are present
        critical_fields = self.CRITICAL_FIELDS.get(domain, [])
        if not critical_fields:
            return 0.5  # Default for unknown domains
        
        present_count = 0
        total_count = len(critical_fields)
        
        if "age" in critical_fields and entities.age is not None:
            present_count += 1
        if "annual_income" in critical_fields and entities.annual_income is not None:
            present_count += 1
        if "state_or_ut" in critical_fields and entities.state_or_ut is not None:
            present_count += 1
        if "category" in critical_fields and entities.category is not None:
            present_count += 1
        if "documents" in critical_fields and entities.documents_mentioned:
            present_count += 1
        
        base_score = present_count / total_count if total_count > 0 else 0.0
        
        # Penalty for high-importance missing items
        penalty = min(missing_count * 0.1, 0.3)
        
        completeness_score = max(0.0, min(1.0, base_score - penalty))
        
        return round(completeness_score, 2)
    
    def _generate_notes(self, missing_info: List[MissingInformation], 
                       completeness_score: float) -> str:
        """Generate explanatory notes"""
        
        high_importance = sum(1 for m in missing_info 
                            if m.importance_level == ImportanceLevel.HIGH.value)
        
        if completeness_score >= 0.7:
            return f"Information is relatively complete ({int(completeness_score*100)}%). Minor gaps can be addressed."
        elif completeness_score >= 0.4:
            return f"Moderate information gaps ({int(completeness_score*100)}% complete). {high_importance} critical fields missing."
        else:
            return f"Severe information gaps ({int(completeness_score*100)}% complete). Cannot provide reliable guidance without {high_importance} critical fields."
    
    def to_json(self, output: UnknownDetectionOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "missing_information": [
                {
                    "unknown_id": m.unknown_id,
                    "field_name": m.field_name,
                    "description": m.description,
                    "importance_level": m.importance_level,
                    "consequence_if_ignored": m.consequence_if_ignored
                }
                for m in output.missing_information
            ],
            "information_completeness_score": output.information_completeness_score,
            "notes": output.notes
        }, indent=2)


# ==================== AGENT 5: TEMPORAL UNCERTAINTY ====================

@dataclass
class TimeDependentFactor:
    factor_id: str
    description: str
    risk_if_outdated: str


@dataclass
class TemporalUncertaintyOutput:
    time_sensitivity_level: str  # RiskLevel
    time_dependent_factors: List[TimeDependentFactor]
    recommended_fresh_checks: List[str]
    notes: str


class TemporalUncertaintyAgent:
    """
    Agent 5: Temporal Uncertainty Agent
    Identifies how time-dependent the answer is (rule changes, deadlines, etc.)
    """
    
    def process(self, user_query: str, query_processor_output: QueryProcessorOutput,
                fact_boundary_output: FactBoundaryOutput,
                unknown_detection_output: UnknownDetectionOutput) -> TemporalUncertaintyOutput:
        """Identify time-related uncertainties"""
        
        time_factors = []
        factor_counter = 1
        
        domain = query_processor_output.detected_domain
        query_lower = user_query.lower()
        
        # Check for explicit time references
        has_current_marker = any(word in query_lower for word in 
                                ['current', 'now', 'today', '2024', '2025', 'latest', 'recent'])
        
        # Domain-specific time sensitivity
        if domain == Domain.GOVERNMENT_SCHEME.value:
            
            time_factors.append(TimeDependentFactor(
                factor_id=f"T{factor_counter}",
                description="Government scheme eligibility criteria may have changed",
                risk_if_outdated="Income limits, age brackets, or eligibility rules frequently updated in budget cycles"
            ))
            factor_counter += 1
            
            time_factors.append(TimeDependentFactor(
                factor_id=f"T{factor_counter}",
                description="Scheme application deadlines may have passed or changed",
                risk_if_outdated="Some schemes have limited enrollment windows or annual cycles"
            ))
            factor_counter += 1
            
            time_factors.append(TimeDependentFactor(
                factor_id=f"T{factor_counter}",
                description="New schemes may have been launched or old ones discontinued",
                risk_if_outdated="Government frequently launches new welfare schemes and phases out old ones"
            ))
            factor_counter += 1
            
            time_sensitivity = RiskLevel.HIGH.value
        
        elif domain == Domain.LEGAL_PRESCREENING.value:
            
            time_factors.append(TimeDependentFactor(
                factor_id=f"T{factor_counter}",
                description="Legal statutes and case law may have evolved",
                risk_if_outdated="Court judgments and amended laws can change legal positions"
            ))
            factor_counter += 1
            
            time_sensitivity = RiskLevel.MEDIUM.value
        
        elif domain == Domain.FINANCIAL_COMPLIANCE.value:
            
            time_factors.append(TimeDependentFactor(
                factor_id=f"T{factor_counter}",
                description="Tax rules and compliance requirements change annually",
                risk_if_outdated="Budget amendments affect tax slabs, deductions, and filing requirements"
            ))
            factor_counter += 1
            
            time_sensitivity = RiskLevel.HIGH.value
        
        else:
            time_sensitivity = RiskLevel.LOW.value
        
        # Generate recommended fresh checks
        recommended_checks = self._generate_fresh_checks(domain, has_current_marker)
        
        # Generate notes
        notes = self._generate_notes(time_sensitivity, time_factors, has_current_marker)
        
        return TemporalUncertaintyOutput(
            time_sensitivity_level=time_sensitivity,
            time_dependent_factors=time_factors,
            recommended_fresh_checks=recommended_checks,
            notes=notes
        )
    
    def _generate_fresh_checks(self, domain: str, has_current_marker: bool) -> List[str]:
        """Generate list of recommended fresh checks"""
        
        checks = []
        
        if domain == Domain.GOVERNMENT_SCHEME.value:
            checks.extend([
                "Check official scheme portal for latest eligibility criteria",
                "Verify scheme is still active and accepting applications",
                "Confirm income limits haven't changed in recent budget",
                "Check state-specific scheme website for local variations"
            ])
        
        elif domain == Domain.LEGAL_PRESCREENING.value:
            checks.extend([
                "Verify current legal provisions (amendments possible)",
                "Check for recent Supreme Court or High Court judgments",
                "Confirm statute of limitations hasn't expired"
            ])
        
        elif domain == Domain.FINANCIAL_COMPLIANCE.value:
            checks.extend([
                "Verify current financial year tax rules",
                "Check latest GST/Income Tax notifications",
                "Confirm compliance deadlines for current period"
            ])
        
        if has_current_marker:
            checks.insert(0, "User specifically asked for current info - verification is CRITICAL")
        
        return checks
    
    def _generate_notes(self, time_sensitivity: str, factors: List[TimeDependentFactor],
                       has_current_marker: bool) -> str:
        """Generate explanatory notes"""
        
        if time_sensitivity == RiskLevel.HIGH.value:
            urgency = "HIGH time-sensitivity. Information could be outdated within months."
        elif time_sensitivity == RiskLevel.MEDIUM.value:
            urgency = "MEDIUM time-sensitivity. Annual verification recommended."
        else:
            urgency = "LOW time-sensitivity. Information is relatively stable."
        
        if has_current_marker:
            urgency += " User explicitly requested current information."
        
        return f"{urgency} {len(factors)} time-dependent factors identified."
    
    def to_json(self, output: TemporalUncertaintyOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "time_sensitivity_level": output.time_sensitivity_level,
            "time_dependent_factors": [
                {
                    "factor_id": f.factor_id,
                    "description": f.description,
                    "risk_if_outdated": f.risk_if_outdated
                }
                for f in output.time_dependent_factors
            ],
            "recommended_fresh_checks": output.recommended_fresh_checks,
            "notes": output.notes
        }, indent=2)


# ==================== AGENT 6: CONFIDENCE CALIBRATION ====================

@dataclass
class EpistemicUncertaintyFactor:
    factor_id: str
    description: str
    severity: str  # RiskLevel


@dataclass
class ConfidenceCalibrationOutput:
    calibrated_confidence: int  # 0-100
    epistemic_uncertainty_factors: List[EpistemicUncertaintyFactor]
    confidence_explanation: str
    safety_flag: str  # SafetyFlag


class ConfidenceCalibrationAgent:
    """
    Agent 6: Confidence Calibration Agent
    Converts council's analysis into calibrated confidence score and safety flag
    """
    
    def process(self, query_processor_output: QueryProcessorOutput,
                fact_boundary_output: FactBoundaryOutput,
                assumption_agent_output: AssumptionAgentOutput,
                unknown_detection_output: UnknownDetectionOutput,
                temporal_agent_output: TemporalUncertaintyOutput) -> ConfidenceCalibrationOutput:
        """Calibrate confidence based on all agent outputs"""
        
        # Start with base confidence
        base_confidence = 80
        
        # Deductions based on agent outputs
        deductions = []
        
        # 1. Missing information penalty (up to 30 points)
        missing_info_penalty = self._calculate_missing_info_penalty(unknown_detection_output)
        base_confidence -= missing_info_penalty
        deductions.append(f"Missing information: -{missing_info_penalty}")
        
        # 2. Assumption risk penalty (up to 20 points)
        assumption_penalty = self._calculate_assumption_penalty(assumption_agent_output)
        base_confidence -= assumption_penalty
        deductions.append(f"Assumption risk: -{assumption_penalty}")
        
        # 3. Time sensitivity penalty (up to 20 points)
        time_penalty = self._calculate_time_penalty(temporal_agent_output)
        base_confidence -= time_penalty
        deductions.append(f"Time sensitivity: -{time_penalty}")
        
        # 4. Clarity boost (up to 10 points)
        if fact_boundary_output.clarity_score >= 0.8:
            base_confidence += 10
            deductions.append(f"High clarity: +10")
        
        # Clamp between 0 and 100
        calibrated_confidence = max(0, min(100, base_confidence))
        
        # Identify epistemic uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(
            assumption_agent_output,
            unknown_detection_output,
            temporal_agent_output
        )
        
        # Generate explanation
        explanation = self._generate_explanation(calibrated_confidence, deductions)
        
        # Determine safety flag
        safety_flag = self._determine_safety_flag(
            calibrated_confidence,
            unknown_detection_output,
            assumption_agent_output
        )
        
        return ConfidenceCalibrationOutput(
            calibrated_confidence=calibrated_confidence,
            epistemic_uncertainty_factors=uncertainty_factors,
            confidence_explanation=explanation,
            safety_flag=safety_flag
        )
    
    def _calculate_missing_info_penalty(self, unknown_output: UnknownDetectionOutput) -> int:
        """Calculate penalty for missing information (0-30 points)"""
        
        completeness = unknown_output.information_completeness_score
        
        # Inverse relationship: lower completeness = higher penalty
        base_penalty = int((1.0 - completeness) * 30)
        
        # Extra penalty for high-importance missing items
        high_importance_count = sum(
            1 for m in unknown_output.missing_information
            if m.importance_level == ImportanceLevel.HIGH.value
        )
        
        extra_penalty = min(high_importance_count * 5, 15)
        
        return min(base_penalty + extra_penalty, 30)
    
    def _calculate_assumption_penalty(self, assumption_output: AssumptionAgentOutput) -> int:
        """Calculate penalty for assumption risk (0-20 points)"""
        
        risk_level = assumption_output.overall_assumption_risk
        
        if risk_level == RiskLevel.HIGH.value:
            return 20
        elif risk_level == RiskLevel.MEDIUM.value:
            return 12
        else:
            return 5
    
    def _calculate_time_penalty(self, temporal_output: TemporalUncertaintyOutput) -> int:
        """Calculate penalty for time sensitivity (0-20 points)"""
        
        sensitivity = temporal_output.time_sensitivity_level
        
        if sensitivity == RiskLevel.HIGH.value:
            return 20
        elif sensitivity == RiskLevel.MEDIUM.value:
            return 10
        else:
            return 3
    
    def _identify_uncertainty_factors(self, assumption_output: AssumptionAgentOutput,
                                     unknown_output: UnknownDetectionOutput,
                                     temporal_output: TemporalUncertaintyOutput) -> List[EpistemicUncertaintyFactor]:
        """Identify key epistemic uncertainty factors"""
        
        factors = []
        factor_counter = 1
        
        # High-risk assumptions
        high_risk_assumptions = [
            a for a in assumption_output.assumptions
            if a.risk_level == RiskLevel.HIGH.value
        ]
        
        if high_risk_assumptions:
            factors.append(EpistemicUncertaintyFactor(
                factor_id=f"E{factor_counter}",
                description=f"{len(high_risk_assumptions)} high-risk assumptions made without verification",
                severity=RiskLevel.HIGH.value
            ))
            factor_counter += 1
        
        # Critical missing information
        critical_missing = [
            m for m in unknown_output.missing_information
            if m.importance_level == ImportanceLevel.HIGH.value
        ]
        
        if critical_missing:
            factors.append(EpistemicUncertaintyFactor(
                factor_id=f"E{factor_counter}",
                description=f"{len(critical_missing)} critical fields missing from user query",
                severity=RiskLevel.HIGH.value
            ))
            factor_counter += 1
        
        # Time sensitivity
        if temporal_output.time_sensitivity_level == RiskLevel.HIGH.value:
            factors.append(EpistemicUncertaintyFactor(
                factor_id=f"E{factor_counter}",
                description="Answer depends on frequently changing rules/deadlines",
                severity=RiskLevel.HIGH.value
            ))
            factor_counter += 1
        
        return factors
    
    def _generate_explanation(self, confidence: int, deductions: List[str]) -> str:
        """Generate human-readable confidence explanation"""
        
        breakdown = "; ".join(deductions)
        
        if confidence >= 70:
            level = "MODERATE-HIGH confidence"
            guidance = "Answer is likely reliable, but verify key details."
        elif confidence >= 40:
            level = "LOW-MODERATE confidence"
            guidance = "Answer has significant uncertainty. Proceed with caution."
        else:
            level = "VERY LOW confidence"
            guidance = "Too much uncertainty to provide reliable guidance."
        
        return f"{level} ({confidence}/100). Breakdown: {breakdown}. {guidance}"
    
    def _determine_safety_flag(self, confidence: int, 
                               unknown_output: UnknownDetectionOutput,
                               assumption_output: AssumptionAgentOutput) -> str:
        """Determine safety flag based on confidence and risks"""
        
        critical_missing = sum(
            1 for m in unknown_output.missing_information
            if m.importance_level == ImportanceLevel.HIGH.value
        )
        
        high_risk_assumptions = sum(
            1 for a in assumption_output.assumptions
            if a.risk_level == RiskLevel.HIGH.value
        )
        
        # Unsafe to answer if:
        # - Confidence very low (<30)
        # - 3+ critical fields missing
        # - 3+ high-risk assumptions
        if confidence < 30 or critical_missing >= 3 or high_risk_assumptions >= 3:
            return SafetyFlag.UNSAFE_TO_ANSWER.value
        
        # Answer with caution if:
        # - Confidence moderate (30-60)
        # - 1-2 critical missing or high-risk assumptions
        elif confidence < 60 or critical_missing >= 1 or high_risk_assumptions >= 1:
            return SafetyFlag.ANSWER_WITH_CAUTION.value
        
        # Safe to answer if confidence high and low risks
        else:
            return SafetyFlag.SAFE_TO_ANSWER.value
    
    def to_json(self, output: ConfidenceCalibrationOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "calibrated_confidence": output.calibrated_confidence,
            "epistemic_uncertainty_factors": [
                {
                    "factor_id": f.factor_id,
                    "description": f.description,
                    "severity": f.severity
                }
                for f in output.epistemic_uncertainty_factors
            ],
            "confidence_explanation": output.confidence_explanation,
            "safety_flag": output.safety_flag
        }, indent=2)


# ==================== AGENT 7: DECISION GUIDANCE ====================

@dataclass
class DecisionGuidanceOutput:
    final_answer_style: str  # AnswerStyle
    user_friendly_summary: str
    explicit_knowns: List[str]
    explicit_unknowns: List[str]
    assumptions_highlighted: List[str]
    calibrated_confidence: int
    safety_flag: str
    recommended_next_steps: List[str]


class DecisionGuidanceAgent:
    """
    Agent 7: Decision Guidance Agent
    Converts council analysis into clear, actionable user guidance
    """
    
    def process(self, user_query: str,
                query_processor_output: QueryProcessorOutput,
                fact_boundary_output: FactBoundaryOutput,
                assumption_agent_output: AssumptionAgentOutput,
                unknown_detection_output: UnknownDetectionOutput,
                temporal_agent_output: TemporalUncertaintyOutput,
                confidence_agent_output: ConfidenceCalibrationOutput) -> DecisionGuidanceOutput:
        """Generate final decision guidance for user"""
        
        safety_flag = confidence_agent_output.safety_flag
        
        # Determine answer style based on safety flag
        if safety_flag == SafetyFlag.UNSAFE_TO_ANSWER.value:
            answer_style = AnswerStyle.NO_DIRECT_DECISION.value
        elif safety_flag == SafetyFlag.ANSWER_WITH_CAUTION.value:
            answer_style = AnswerStyle.CAUTIOUS_TENTATIVE_DECISION.value
        else:
            answer_style = AnswerStyle.DIRECT_DECISION.value
        
        # Generate user-friendly summary
        summary = self._generate_summary(
            user_query, answer_style, confidence_agent_output, query_processor_output
        )
        
        # Extract explicit knowns
        knowns = [fact.description for fact in fact_boundary_output.known_facts]
        
        # Extract explicit unknowns
        unknowns = [
            f"{m.field_name}: {m.description}"
            for m in unknown_detection_output.missing_information
        ]
        
        # Extract assumptions
        assumptions = [a.description for a in assumption_agent_output.assumptions]
        
        # Generate next steps
        next_steps = self._generate_next_steps(
            answer_style,
            unknown_detection_output,
            temporal_agent_output,
            query_processor_output
        )
        
        return DecisionGuidanceOutput(
            final_answer_style=answer_style,
            user_friendly_summary=summary,
            explicit_knowns=knowns,
            explicit_unknowns=unknowns,
            assumptions_highlighted=assumptions,
            calibrated_confidence=confidence_agent_output.calibrated_confidence,
            safety_flag=safety_flag,
            recommended_next_steps=next_steps
        )
    
    def _generate_summary(self, user_query: str, answer_style: str,
                         confidence_output: ConfidenceCalibrationOutput,
                         qp_output: QueryProcessorOutput) -> str:
        """Generate user-friendly summary"""
        
        confidence = confidence_output.calibrated_confidence
        domain = qp_output.detected_domain
        
        if answer_style == AnswerStyle.NO_DIRECT_DECISION.value:
            return (
                f"⚠️ I cannot provide a reliable answer to your {domain.replace('_', ' ')} query "
                f"due to insufficient information (confidence: {confidence}%). "
                f"Too many critical details are missing or uncertain. "
                f"Please see the unknowns and next steps below to gather necessary information first."
            )
        
        elif answer_style == AnswerStyle.CAUTIOUS_TENTATIVE_DECISION.value:
            return (
                f"⚡ Based on the limited information provided, I can offer a tentative assessment "
                f"for your {domain.replace('_', ' ')} query (confidence: {confidence}%). "
                f"However, several assumptions have been made and important details are missing. "
                f"Please review the assumptions and unknowns carefully before proceeding."
            )
        
        else:  # DIRECT_DECISION
            return (
                f"✅ I can provide a reasonably confident answer to your {domain.replace('_', ' ')} query "
                f"(confidence: {confidence}%). "
                f"Based on the information you've provided, here's my assessment. "
                f"Still, please verify key details from official sources."
            )
    
    def _generate_next_steps(self, answer_style: str,
                            unknown_output: UnknownDetectionOutput,
                            temporal_output: TemporalUncertaintyOutput,
                            qp_output: QueryProcessorOutput) -> List[str]:
        """Generate recommended next steps"""
        
        steps = []
        
        # Add steps based on missing critical information
        critical_missing = [
            m for m in unknown_output.missing_information
            if m.importance_level == ImportanceLevel.HIGH.value
        ]
        
        for missing in critical_missing[:3]:  # Top 3 critical items
            if "income" in missing.field_name.lower():
                steps.append("Obtain an official income certificate from the competent authority")
            elif "age" in missing.field_name.lower():
                steps.append("Provide your current age or date of birth")
            elif "state" in missing.field_name.lower():
                steps.append("Specify your state or union territory of residence")
            elif "category" in missing.field_name.lower():
                steps.append("Clarify your caste category and obtain category certificate if applicable")
            elif "document" in missing.field_name.lower():
                steps.append("Gather required documents (Aadhaar, domicile certificate, etc.)")
            else:
                steps.append(f"Clarify: {missing.description}")
        
        # Add temporal verification steps
        if temporal_output.time_sensitivity_level in [RiskLevel.HIGH.value, RiskLevel.MEDIUM.value]:
            steps.extend(temporal_output.recommended_fresh_checks[:2])
        
        # Add domain-specific steps
        if qp_output.detected_domain == Domain.GOVERNMENT_SCHEME.value:
            steps.append("Visit the official scheme portal or nearest government office for latest information")
            steps.append("Consult with a local NGO or scheme facilitator for application assistance")
        
        # Ensure we have at least some steps
        if not steps:
            steps.append("Verify information with official sources")
            steps.append("Consult with a domain expert if proceeding with important decisions")
        
        return steps[:5]  # Return max 5 steps
    
    def to_json(self, output: DecisionGuidanceOutput) -> str:
        """Convert output to JSON"""
        return json.dumps({
            "final_answer_style": output.final_answer_style,
            "user_friendly_summary": output.user_friendly_summary,
            "explicit_knowns": output.explicit_knowns,
            "explicit_unknowns": output.explicit_unknowns,
            "assumptions_highlighted": output.assumptions_highlighted,
            "calibrated_confidence": output.calibrated_confidence,
            "safety_flag": output.safety_flag,
            "recommended_next_steps": output.recommended_next_steps
        }, indent=2)


# ==================== MAIN UNCERTAINTY-FIRST AGENT COUNCIL ====================

class UncertaintyFirstAgentCouncil:
    """
    Main orchestrator for the Uncertainty-First Agent Council
    Coordinates all 7 agents in sequence
    """
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.fact_boundary_agent = FactBoundaryAgent()
        self.assumption_agent = AssumptionAgent()
        self.unknown_detection_agent = UnknownDetectionAgent()
        self.temporal_agent = TemporalUncertaintyAgent()
        self.confidence_agent = ConfidenceCalibrationAgent()
        self.decision_guidance_agent = DecisionGuidanceAgent()
    
    def process_query(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process user query through all 7 agents
        
        Args:
            user_query: User's natural language question
            verbose: If True, print intermediate outputs
        
        Returns:
            Dictionary with all agent outputs
        """
        
        print("\n" + "="*80)
        print("UNCERTAINTY-FIRST AGENT COUNCIL")
        print("="*80)
        print(f"\nUser Query: {user_query}\n")
        
        # Agent 1: Query Processor
        print("🔄 Agent 1: Query Processor...")
        qp_output = self.query_processor.process(user_query)
        if verbose:
            print(self.query_processor.to_json(qp_output))
        print("✅ Complete\n")
        
        # Agent 2: Fact Boundary
        print("🔄 Agent 2: Fact Boundary Agent...")
        fb_output = self.fact_boundary_agent.process(user_query, qp_output)
        if verbose:
            print(self.fact_boundary_agent.to_json(fb_output))
        print("✅ Complete\n")
        
        # Agent 3: Assumption Agent
        print("🔄 Agent 3: Assumption Agent...")
        aa_output = self.assumption_agent.process(user_query, qp_output, fb_output)
        if verbose:
            print(self.assumption_agent.to_json(aa_output))
        print("✅ Complete\n")
        
        # Agent 4: Unknown Detection
        print("🔄 Agent 4: Unknown Detection Agent...")
        ud_output = self.unknown_detection_agent.process(user_query, qp_output, fb_output, aa_output)
        if verbose:
            print(self.unknown_detection_agent.to_json(ud_output))
        print("✅ Complete\n")
        
        # Agent 5: Temporal Uncertainty
        print("🔄 Agent 5: Temporal Uncertainty Agent...")
        tu_output = self.temporal_agent.process(user_query, qp_output, fb_output, ud_output)
        if verbose:
            print(self.temporal_agent.to_json(tu_output))
        print("✅ Complete\n")
        
        # Agent 6: Confidence Calibration
        print("🔄 Agent 6: Confidence Calibration Agent...")
        cc_output = self.confidence_agent.process(qp_output, fb_output, aa_output, ud_output, tu_output)
        if verbose:
            print(self.confidence_agent.to_json(cc_output))
        print("✅ Complete\n")
        
        # Agent 7: Decision Guidance
        print("🔄 Agent 7: Decision Guidance Agent...")
        dg_output = self.decision_guidance_agent.process(
            user_query, qp_output, fb_output, aa_output, ud_output, tu_output, cc_output
        )
        if verbose:
            print(self.decision_guidance_agent.to_json(dg_output))
        print("✅ Complete\n")
        
        print("="*80)
        print("COUNCIL PROCESSING COMPLETE")
        print("="*80)
        
        return {
            "query_processor": qp_output,
            "fact_boundary": fb_output,
            "assumption_agent": aa_output,
            "unknown_detection": ud_output,
            "temporal_uncertainty": tu_output,
            "confidence_calibration": cc_output,
            "decision_guidance": dg_output
        }
    
    def get_user_response(self, user_query: str) -> str:
        """
        Get final user-facing response
        
        Args:
            user_query: User's question
        
        Returns:
            Formatted user response string
        """
        
        results = self.process_query(user_query, verbose=False)
        dg = results["decision_guidance"]
        
        response = []
        response.append("\n" + "="*80)
        response.append("RESPONSE TO YOUR QUERY")
        response.append("="*80 + "\n")
        
        response.append(dg.user_friendly_summary)
        response.append("\n")
        
        response.append(f"📊 Confidence Level: {dg.calibrated_confidence}%")
        response.append(f"🚦 Safety Assessment: {dg.safety_flag.replace('_', ' ').title()}\n")
        
        if dg.explicit_knowns:
            response.append("✅ WHAT WE KNOW:")
            for i, known in enumerate(dg.explicit_knowns, 1):
                response.append(f"   {i}. {known}")
            response.append("")
        
        if dg.explicit_unknowns:
            response.append("❓ WHAT WE DON'T KNOW:")
            for i, unknown in enumerate(dg.explicit_unknowns, 1):
                response.append(f"   {i}. {unknown}")
            response.append("")
        
        if dg.assumptions_highlighted:
            response.append("⚠️  ASSUMPTIONS WE'RE MAKING:")
            for i, assumption in enumerate(dg.assumptions_highlighted, 1):
                response.append(f"   {i}. {assumption}")
            response.append("")
        
        if dg.recommended_next_steps:
            response.append("📋 RECOMMENDED NEXT STEPS:")
            for i, step in enumerate(dg.recommended_next_steps, 1):
                response.append(f"   {i}. {step}")
            response.append("")
        
        response.append("="*80)
        
        return "\n".join(response)


# ==================== DEMO & TESTING ====================

def run_demo():
    """Run demonstration with sample queries"""
    
    council = UncertaintyFirstAgentCouncil()
    
    # Sample queries for different scenarios
    test_queries = [
        # High clarity query
        "I am 25 years old from Maharashtra with annual income of 3 LPA and belong to OBC category. Am I eligible for PM-KISAN scheme?",
        
        # Medium clarity query
        "I earn around 20000 per month. Can I apply for some government scholarship?",
        
        # Low clarity query
        "Am I eligible for any schemes?",
        
        # Time-sensitive query
        "What are the current tax benefits for senior citizens in 2025?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*80}")
        print(f"TEST QUERY {i}")
        print(f"{'#'*80}")
        
        response = council.get_user_response(query)
        print(response)
        
        input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              UNCERTAINTY-FIRST AGENT COUNCIL                                 ║
║              An Agentic AI System for Explicit Uncertainty Modeling          ║
║                                                                              ║
║              Project Based Learning - Team                                   ║
║              Amitkumar Racha | Bontha Mallikarjun Reddy | Neil Cardoz       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    run_demo()
