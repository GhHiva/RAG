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
  1. `Naive RAG` : No complex data embeding and indexing. Ex: augment the user's input. (keywords)
  2. `Advanced RAG` : complex data embeding(vector search) and indexing. (multiple data types, multimodal data, structured or unstructured) => complex scenarios
  3. `Modular RAG` : any scenario + naive RAG + advanced RAG + ML, and any algorithm needed to complete a complex project.

- RAG framework contains two main parts:
    1. a retriever : can be any frameworks such as Activeloop, Pinecone, ...
    2. a generator : can be any LLM such as Gemini, GPT-4o,...

_________________________
# RAG vs Fine-tuning:
RAG and fine-tuning are not always interchangeable. The more data in RAG datasets, the more cumbersome we have. On the other hand, we cannot fine-tune a model with dynamic and ever-changing data!
* The decision of whether to implement RAG or fine-tune a model depends on the proportion of `parametric` vs `non-parametric` info!
**Parametric**:
  - refers to the model's parameters `weights` which are learned through training data.
  - model's knowledge is `stored` in these learned `weights` and `biases`.
  - the original training data is transformed into mathematical form.
  - the model remember what it learned from data but the data itself is not stored explicitly.
**Non-Parametric**:
  - stores explicit data that can be accessed directly.
  - data remains available.
* Difference between RAG and fine-tuning relies on the amount of `static (parametric)` and `dynamic (non-parametric)` data that then model must process.
* `Decision-making threshhold`: Before building a system that searches documents (RAG), the AI project manager should first evaluate whether the existing trained AI model alone (without extra data) is good enough for the task.

# The RAG ecosystem:

RAG-driven generative AI is a framework that can be implemented in many configurations.
It contains four parts:
- Data: where is the data coming from? is it reliable? sufficient? what about the privacy?
- Storage: how to store before or after the processing? how much we wanna store?
- Retrieval: how the correct data retrieved before the generative model use it? what type of RAG framework will be successful for a project?
- Generation: which generative AI model fit into the chosen RAG framework?


**The data, storage, and generation domains depend heavily on the type of RAG framwork you choose.**
![‎New Note ‎1](https://github.com/user-attachments/assets/07e368d1-b6e9-427c-9063-1b9c7d142893)

# Foundation and Basic Implementation:
1. Environment: set up for openAI API integration
2. Generator: create a function for the generator (using GPT-4o), define a function to print a formatted response
3. Data: set up with the list of documents
4. Query: for user input

## Environment:
- install openai, (openai==version).
- create a api key and store it in a secret file.

## Generator:
- generate content and time (to measure the time the requests take)
- create a function that creates a prompt(with instruction and the user input)

## Data:
- wrap data in one string.

## Query:
- the retriever query process depends on how the data was processed, the query itself is simply `user input` or `automated input from another AI agent`.
- main query is the junction between the retriever and the generator.

### Retrieval metrics:
`Calculate_cosine_similarity(text1,text2)`:
measures the cosine of the angle between the user query and each document in a corpus. But it has limitations when dealing with ambiguous queries. The low score occures cuz the mathematical model lacks contextual understanding to differentiate between the different meaning of the word if it had. 
**Enhanced similarity**
.
.
.
.
`Calculate_enhanced_similarity(text1,text2)`:






