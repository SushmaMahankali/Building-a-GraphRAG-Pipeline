# Building-a-GraphRAG-Pipeline

I'm going to build a pipeline that:

Reads text.
Extracts relationships (Subject -> Predicate -> Object).
Builds a Graph using NetworkX.
Retrieves context by walking the graph (Multi-hop reasoning).
Answers a question based on that deep context.

A few libraries has to be installed
pip install networkx langchain langchain-ollama

Step 1: Loading the LLM
We need an LLM to do the main work, like reading text and pulling out logic. Here, we’ll use Ollama to run Mistral on your own machine. It’s quick, free, and works well for reasoning tasks.
For production, i have use GPT-4o or Claude 3.5 Sonnet for better accuracy. But for learning, Mistral is a great choice.

Step 2: Turning Text into Data
This step is the most important. We can’t put raw text straight into a graph; we need triples. A triple is the basic unit of a knowledge graph: (Head) -> [Relation] -> (Tail). We’ll use a StrOutputParser to make sure the LLM gives us clean.

Step 3: The Data Source
Let’s test our system with a short example about the AI industry. The facts are in separate sentences, and my goal is to connect them.

Step 4: Building the Graph
Now we’ll use NetworkX, a Python library for working with graphs. We’ll take the JSON triples from Step 3 and actually create the connections.

Step 5: Multi-Hop
This is where smart retrieval happens. In standard RAG, searching for “ChatGPT” gives you the sentence “ChatGPT is used by millions.” With GraphRAG, we start at “ChatGPT” and explore its connections:

Start at ChatGPT.
Look backward: “Powered by GPT-4”.
Walk to GPT-4: “Developed by OpenAI”.
Walk to OpenAI: “Invested in by Microsoft”.
Now we can see that Microsoft is linked to ChatGPT, even though they were never mentioned together in the same sentence in the original text.

Step 6: The Final Answer
Finally, we feed that rich, interconnected context back to the LLM to answer the user’s question.
The model will correctly identify Microsoft. This works because the graph context includes the chain: Microsoft -> invested -> OpenAI -> developed -> GPT-4 -> powers -> ChatGPT.

