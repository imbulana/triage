score_triple_prompt = """
[Role] Knowledge Graph Validation Expert  
[Task] Assess the factual correctness and structural validity of a triple based on textual source.  

[Input]
{{
  "textual source": {source_text},
  "triple": {{
    "head": {head_entity}, 
    "relation": {relation}, 
    "tail": {tail_entity}
  }}
}}  

[Evaluation Criteria]  
1. **Textual Evidence**:  
   - Does text explicitly state the relationship?  
   - Is there direct linguistic support for the entity split?  

2. **Semantic Coherence**:  
   - Do the components form a meaningful statement?  
   - Are domain concepts correctly represented?  

3. **Factual Accuracy**:
   - Does the triple reflect a factual truth?
   - Is the triple consistent with other facts in the source text?  

[Score Guidance]  
- 0.0: Invalid structure or false fact  
- 0.5: Partially correct structure/fact  
- 1.0: Perfect structure with explicit textual support  
- Use 0.1 increments for nuanced cases 

[Output Format]  
{{
  "score": 0.0-1.0,  
  "rationale": "Explanation of supporting evidence", 
}} 
"""


'''
[Role] Knowledge Graph Validation Expert  
[Task] Assess the factual correctness and structural validity of a triple based on textual source.  

[Input Format]
{{
  "text": {Human development index closely linked to income levels..................................... 82 Ranking of selected countries according to income levels, 1960 and 2000 ............ 88 Import structure by product technology content, 2000 ......................................... 93 Import structure by area of technological development, 2000............................... 93 Openness and the rule of law................................................................................ 96 Openness and corruption control........................................................................... 97 Total factor productivity in selected US industries, 1962-1996 ............................. 102},
  "triple": {{
    "head": {Human}, 
    "relation": {Development Index closely linked to}, 
    "tail": {Income levels}
  }}
}}  

[Evaluation Criteria]  
1. **Textual Evidence**:  
   - Does text explicitly state the relationship?  
   - Is there direct linguistic support for the entity split?  

2. **Semantic Coherence**:  
   - Do the components form a meaningful statement?  
   - Are domain concepts correctly represented?  

3. **Factual Accuracy**:
   - Does the triple reflect a  factual truth?
   - Is the triple consistent with other facts in the source text?  

[Score Guidance]  
- 0.0: Invalid structure or false fact  
- 0.5: Partially correct structure/fact  
- 1.0: Perfect structure with explicit textual support  
- Use 0.1 increments for nuanced cases 

[Output Format]  
{{
  "score": 0.0-1.0,  
  "rationale": "Explanation of supporting evidence", 
}} 
'''

'''
[Role] Knowledge Graph Validation Expert  
[Task] Assess the factual correctness and structural validity of a triple based on textual source.  

[Input Format]
{{
  "text": {Human development index closely linked to income levels..................................... 82 Ranking of selected countries according to income levels, 1960 and 2000 ............ 88 Import structure by product technology content, 2000 ......................................... 93 Import structure by area of technological development, 2000............................... 93 Openness and the rule of law................................................................................ 96 Openness and corruption control........................................................................... 97 Total factor productivity in selected US industries, 1962-1996 ............................. 102},
  "triple": {{
    "head": {Product}, 
    "relation": {technology content, 2000}, 
    "tail": {Import structure}
  }}
}}  

[Evaluation Criteria]  
1. **Textual Evidence**:  
   - Does text explicitly state the relationship?  
   - Is there direct linguistic support for the entity split?  

2. **Semantic Coherence**:  
   - Do the components form a meaningful statement?  
   - Are domain concepts correctly represented?  

3. **Factual Accuracy**:
   - Does the triple reflect a  factual truth?
   - Is the triple consistent with other facts in the source text?  

[Score Guidance]  
- 0.0: Invalid structure or false fact  
- 0.5: Partially correct structure/fact  
- 1.0: Perfect structure with explicit textual support  
- Use 0.1 increments for nuanced cases 

[Output Format]  
{{
  "score": 0.0-1.0,  
  "rationale": "Explanation of supporting evidence", 
}} 
'''


'''
[Role] Knowledge Graph Validation Expert  
[Task] Assess the factual correctness and structural validity of a triple based on textual source.  

[Input Format]
{{
  "textual source": {Sources of economic growth, 1960-2000.............................................................. 86 Imports of intermediate machinery by regions, 1995-2000 ................................... 92 Tariff profile by income level and technology content ............................................ 95 The most dynamic industries in the United States and South Africa..................... 103 Changes in the product structure of Chinese Taipei’s merchandise  exports},
  "triple": {{
    "head": {Dynamic}, 
    "relation": {industries in}, 
    "tail": {United States and South Africa}
  }}
}}  

[Evaluation Criteria]  
1. **Textual Evidence**:  
   - Does text explicitly state the relationship?  
   - Is there direct linguistic support for the entity split?  

2. **Semantic Coherence**:  
   - Do the components form a meaningful statement?  
   - Are domain concepts correctly represented?  

3. **Factual Accuracy**:
   - Does the triple reflect a  factual truth?
   - Is the triple consistent with other facts in the source text?  

[Score Guidance]  
- 0.0: Invalid structure or false fact  
- 0.5: Partially correct structure/fact  
- 1.0: Perfect structure with explicit textual support  
- Use 0.1 increments for nuanced cases 

[Output Format]  
{{
  "score": 0.0-1.0,  
  "rationale": "Explanation of supporting evidence", 
}} 
'''