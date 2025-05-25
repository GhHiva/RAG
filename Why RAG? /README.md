* `Generative AI models` can only generate responses based on the data that they have been trained on! It means, they do not have the access to the external and uptodate data.
  
* `Retrival Augmented Generation (RAG)` is a framework that helps to improve the model output accuracy by combining retrival relevant data from the external sources in real time.

* `Generative AI platforms`: Hugging Face, Google Vertex AI, OpenAI, LangChain, and more.
* `RAG frameworks`: Pinecone, Chroma, Activeloop, LIamaIndex, and so on.

* RAG can only exist whithin an ecosystem.

![‎Question ‎1](https://github.com/user-attachments/assets/2f17844b-426a-4589-bca3-9a767857341a)

- Python pipline for an entry-level naive RAG with keyword search and matching.
- Python pipline for an advanced RAG system with vector search and index-based retrieval.
- Build modular RAG that takes both Naive and advanced RAG into account.
______________________________________
# Three main RAG configurations:
  1. Naive RAG : No complex data embeding and indexing. Ex: augment the user's input. (keywords)
  2. Advanced RAG : complex data embeding(vector search) and indexing. (multiple data types, multimodal data, structured or unstructured) => complex scenarios
  3. Modular RAG : any scenario + naive RAG + advanced RAG + ML, and any algorithm needed to complete a complex project.

- RAG framework contains two main parts:
    1. a retriever : can be any frameworks such as Activeloop, Pinecone, ...
    2. a generator : can be any LLM such as Gemini, GPT-4o,...

_________________________
# RAG vs Fine-tuning
