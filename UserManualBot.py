# Import necessary modules
from typing import List

import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import YoutubeLoader

@st.cache_data
def youtubeLinkToDocs(youtubeLink: str) -> List[Document]:
    print(f'youtubeLink========={youtubeLink}')
    loader = YoutubeLoader.from_youtube_url(youtubeLink,add_video_info=True)
    docs = loader.load()
    doc_chunks = []

    for doc in docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"title": doc.metadata["title"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['title']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
        print(doc_chunks)
    return doc_chunks

@st.cache_data
def embedding(youtubeLink):
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(documents=pages, embedding=embeddings)
        index.save_local(f"faiss_index")

    st.success("Embeddings done.", icon="✅")
    print(f'=============={index}')
    return index


# Set up the Streamlit app
st.title("Chatbot with Youtube Transcript ")
st.markdown(
    """ 
        ####  Based on your youtube video transcript, you can ask any question 
        ----
        """
)


# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Enter your youtube link you want to talk with
    2. Enter Your OpenAI api key
    3. Ask question

    **Note : API key not be stored.**
    """
)

youtube_Link = st.text_input('Youtube Link')

if youtube_Link:
    pages = youtubeLinkToDocs(youtube_Link)
    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
        api = st.text_input(
            "**Enter OpenAI API Key**",
            type="password",
            placeholder="sk-",
            help="https://platform.openai.com/account/api-keys",
        )
        if api:
            index = embedding(youtube_Link)
            # index = FAISS.load_local("faiss_index", embeddings = OpenAIEmbeddings(openai_api_key=api)) 

            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=api),
                chain_type = "map_reduce",
                retriever=index.as_retriever(),
            )
            tools = [
                Tool(
                    name="State of Union QA System",
                    func=qa.run,
                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                )
            ]
            prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                        You have access to a single tool:"""
            suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""
            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )
            
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )
                
            llm_chain = LLMChain(
                llm=OpenAI(
                    temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                ),
                prompt=prompt,
            )
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything ",
            )
            # https://www.youtube.com/watch?v=d7vfUodP0c4&t=40s
            # https://www.youtube.com/watch?v=HFPQ5zpgbsg
            
            if query:
                with st.spinner(
                    "Generating answer : `{}` ".format(query)
                ):
                    res = agent_chain.run(query)
                    st.info(res, icon=None)
            with st.expander("History/Memory"):
                st.session_state.memory




