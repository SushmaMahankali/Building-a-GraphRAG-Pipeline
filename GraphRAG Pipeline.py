import networkx as nx
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ======================================================
# 1. Load Local LLM (VERY SMALL MODEL)
# ======================================================

llm = ChatOllama(
    model="qwen2.5:0.5b",   # works on 4GB RAM
    temperature=0
)


# ======================================================
# 2. JSON Extraction Helper (CRITICAL)
# ======================================================

def extract_json(text: str):
    """
    Safely extract JSON array from LLM output.
    Handles extra text, markdown, or explanations.
    """
    if not text:
        raise ValueError("LLM returned empty output")

    match = re.search(r"\[.*\]", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found in output:\n{text}")

    return json.loads(match.group(0))


# ======================================================
# 3. Prompt for Knowledge Graph Extraction
# ======================================================

extract_prompt = PromptTemplate(
    template="""
You are an information extraction system.

Extract factual knowledge triples from the text.

Return ONLY a JSON array.
No explanation.
No markdown.
No comments.

Example:
[
  {{
    "head": "A",
    "relation": "related_to",
    "tail": "B"
  }}
]

Text:
{text}
""",
    input_variables=["text"],
)

raw_chain = extract_prompt | llm | StrOutputParser()


# ======================================================
# 4. Example Input Data
# ======================================================

company_text = """
OpenAI was founded by Sam Altman and Elon Musk.
OpenAI developed GPT-4.
GPT-4 powers ChatGPT.
Microsoft partnered with OpenAI.
Microsoft invested 10 billion dollars in OpenAI.
ChatGPT is used by millions of users worldwide.
"""


# ======================================================
# 5. Extract Knowledge Graph Triples
# ======================================================

print("\nExtracting knowledge graph triples...\n")

raw_output = raw_chain.invoke({"text": company_text})

triples = extract_json(raw_output)

print("Extracted Triples:\n")
for t in triples:
    print(t)


# ======================================================
# 6. Build Knowledge Graph
# ======================================================

kg = nx.DiGraph()

def build_knowledge_graph(triples):
    for item in triples:
        head = item.get("head")
        tail = item.get("tail")
        relation = item.get("relation")

        if head and tail and relation:
            kg.add_node(head)
            kg.add_node(tail)
            kg.add_edge(head, tail, label=relation)

build_knowledge_graph(triples)


print("\nNodes in Graph:")
print(list(kg.nodes()))

print("\nEdges in Graph:")
for h, t, d in kg.edges(data=True):
    print(f"{h} --[{d['label']}]--> {t}")


# ======================================================
# 7. Multi-Hop Graph Retrieval
# ======================================================

def retrieve_graph_context(entity, max_depth=3):
    context = set()
    visited = set()

    def dfs(node, depth):
        if depth > max_depth:
            return

        visited.add(node)

        # outgoing
        for neighbor in kg.successors(node):
            relation = kg[node][neighbor]["label"]
            context.add(f"{node} {relation} {neighbor}")
            if neighbor not in visited:
                dfs(neighbor, depth + 1)

        # incoming
        for predecessor in kg.predecessors(node):
            relation = kg[predecessor][node]["label"]
            context.add(f"{predecessor} {relation} {node}")
            if predecessor not in visited:
                dfs(predecessor, depth + 1)

    if entity in kg.nodes:
        dfs(entity, 1)

    return ". ".join(context)


# ======================================================
# 8. Final RAG Prompt
# ======================================================

final_prompt = PromptTemplate(
    template="""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

rag_chain = final_prompt | llm


# ======================================================
# 9. Ask Multi-Hop Question
# ======================================================

entity = "ChatGPT"

graph_context = retrieve_graph_context(entity, max_depth=3)

print("\nRetrieved Graph Context:\n")
print(graph_context)


question = "Which company invested in the company that built ChatGPT?"

response = rag_chain.invoke(
    {
        "context": graph_context,
        "question": question
    }
)

print("\nFinal Answer:\n")
print(response.content)
