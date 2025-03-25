import uuid
import base64
import fitz
import io
import magic

import PIL.Image

import streamlit as st

from dotenv import load_dotenv

import chromadb.api

from unstructured.partition.auto import partition
from unstructured.partition.common import UnsupportedFileFormatError

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from custom_retriever import CustomMultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def mime_type(file):
    """
    Determines the MIME type of a given file.

    Args:
        file: file object, file here is the file directly uploaded through st.file_uploader
    Returns:
        mime_type: str
    """
    file_magic = magic.Magic(mime=True)
    file.seek(0)
    mime_type = file_magic.from_buffer(file.read())
    return mime_type

def isBase64(sb):
    """
    Check if a string is base64 encoded.

    Args:
        sb: str, string to check
    Returns:
        bool: True if the string is base64 encoded, False otherwise
    """
    try:
            if isinstance(sb, str):
                    # If there's any unicode here, an exception will be thrown and the function will return false
                    sb_bytes = bytes(sb, 'ascii')
            elif isinstance(sb, bytes):
                    sb_bytes = sb
            else:
                    raise ValueError("Argument must be string or bytes")
            return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False
        
def get_images_base64(chunks):
    """
    Extract images from a list of chunks.
    
    Args:
        chunks: list, list of chunks
    Returns:
        images_b64: list, list of images, in base64 format
    """
    images_b64 = []
    for chunk in chunks:
        if chunk.category == "CompositeElement":
            chunk_els = chunk.metadata.orig_elements
        for el in chunk_els:
            if "Image" in str(type(el)):
                images_b64.append(el.metadata.image_base64)
    return images_b64

def display_base64_image(base64_code):
    """
    Display a base64 encoded image in Streamlit.

    Args:
        base64_code: str, base64 encoded image
    Returns:
        None, just directly displays the image in Streamlit interface
    """
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Convert the binary data to an image
    image = PIL.Image.open(io.BytesIO(image_data))

    # Display the image in Streamlit
    st.image(image)

def summarize(texts, tables, images):
    """
    Summarize texts, tables, and images using gpt-4o-mini model.

    Args:
        texts: list, list of text chunks (CompositeElement)
        tables: list, list of table chunks (TableElement)
        images: list, list of images in base64 format
    Returns:
        text_summaries: list, list of text summaries
        table_summaries: list, list of table summaries
        image_summaries: list, list of image summaries
    """

    prompt_text = """
    You are an assistant tasked with summarizing tables and texts.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Chain for summarizing text and tables
    summarize_chain = {"element": lambda x: x} | prompt | ChatOpenAI(temperature=0.5, model="gpt-4o-mini") | StrOutputParser()

    # Summarize text
    with get_openai_callback() as cb:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
        print(f"Text Summaries Cost:\n\n{cb}\n\n")

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables] # parsed to the model as html as it allows the model to better understand the structure of the table for additional context
    with get_openai_callback() as cb:
        table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
        print(f"Table Summaries Cost:\n\n{cb}\n\n")


    # Summarize images
    prompt_image = """
    Describe the image in detail. 
    Highlight keywords and relevant information such that this description can be referenced for further analysis regarding the image
    """
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_image},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    summarize_chain_images = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

    with get_openai_callback() as cb:
        image_summaries = summarize_chain_images.batch(images)
        print(f"Image Summaries Cost:\n\n{cb}\n\n")

    return text_summaries, table_summaries, image_summaries

def display_chunk_pages(file, chunk):
    """
    Display the pages of a chunk in a PDF file. 
    Works with render_page, plot_pdf_with_boxes, extract_page_number_from_chunk and display_base64_image functions.
    
    Args:
        file: file object, file here is the file directly uploaded through st.file_uploader
        chunk: chunk, chunk to display
    Returns:
        None, just directly displays the pages in Streamlit interface
    """
    try:
        page_numbers = extract_page_number_from_chunk(chunk)
        
        docs = []
        for element in chunk.metadata.orig_elements:
            metadata = element.metadata.to_dict()
            if "Table" in str(type(element)):
                metadata["category"] = "Table"
            elif "Image" in str(type(element)):
                metadata["category"] = "Image"
            else:
                metadata["category"] = "Text"
        
            docs.append(Document(page_content=element.text, metadata=metadata))
        
        for page_number in page_numbers:
            render_page(file, docs, page_number, False)
    except Exception as e:
        if isBase64(chunk):
            display_base64_image(chunk)
        else:
            st.write(f"Error: {e}")

def extract_page_number_from_chunk(chunk):
    """
    Extract page numbers from a chunk.
    
    Args:
        chunk: chunk, chunk to extract page numbers from
    Returns:
        page_numbers: set, set of page numbers"""
    elements = chunk.metadata.orig_elements

    page_numbers = set()
    for element in elements:
        page_numbers.add(element.metadata.page_number)

    return page_numbers

def render_page(file, doc_list: list, page_number: int, print_text=True) -> None:
    """
    Passes the page and the segments to be plotted out to the plot_pdf_with_boxes function.
    
    Args:
        file: file object, file here is the file directly uploaded through st.file_uploader
        doc_list: list, list of documents
        page_number: int, page number to render
    Returns:
        None
    """
    file.seek(0)
    bytes_stream = file.read()
    pdf_page = fitz.open(stream=bytes_stream).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get("page_number") == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")

def plot_pdf_with_boxes(pdf_page, segments):
    """
    Plots the PDF page with bounding boxes around the specified segments.
    
    Args:
        pdf_page: fitz.Page, PDF page to plot
        segments: list, list of segments to plot
    Returns:
        None, just directly displays the PDF page in Streamlit interface
    """
    pix = pdf_page.get_pixmap()
    pil_image = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height)
            for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(
            scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
        )
        ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)

def parse_docs(docs):
    """
    Split base64-encoded images and texts
    
    Args:
        docs: list, list of documents
    Returns:
        dict: dictionary containing images and texts
    """
    b64 = []
    text = []
    for doc in docs:
        try:
            base64.b64decode(doc['doc'])
            b64.append(doc['doc'])
        except Exception as e:
            text.append(doc['doc'])
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    """
    Build prompt to query the model.
    
    Args:
        kwargs: dict, dictionary containing context and question
    Returns:
        ChatPromptTemplate: prompt to query the model
    """
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    
    try:
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text
    except AttributeError:
        st.write(text_element)

    # construct prompt with context (including images)
    prompt_template = f"""
    Please answer the question using only the context provided, which includes the text, tables, and image below. 
    Do not rely on any external knowledge.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="MMRAG", page_icon="🤖")
    st.header("Multimodal Retrieval Augmented Generative Chatbot")

    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    # print("\nPreviously Processed File: ", st.session_state.processed_file)

    file = st.file_uploader("Upload your file")
    if file is not None:
        # print("\nCurrent File: ",file)

        # Chat with the model
        user_query = st.chat_input("What would you like to know?")

        if user_query:
            with st.chat_message("Human"):
                st.markdown(user_query)
            try:

                if file == st.session_state.processed_file:
                    print("File already summarized and in ChromeDB")
                else:
                    
                    # partitioning the file
                    chunks = partition(
                        file = file,
                        strategy="hi_res",  # mandatory to infer tables
                        
                        extract_image_block_types=["Image"],      
                        extract_image_block_to_payload=True,   # to extract base64 for API usage

                        chunking_strategy="by_title",          
                        max_characters=10000,                  
                        combine_text_under_n_chars=2000,       
                        new_after_n_chars=6000,
                    )

                    texts = [chunk for chunk in chunks if chunk.category == "CompositeElement"]
                    tables = [chunk for chunk in chunks if chunk.category == "Table"]
                    images = get_images_base64(chunks)

                    text_summaries, table_summaries, image_summaries = summarize(texts, tables, images)
                    
                    chromadb.api.client.SharedSystemClient.clear_system_cache()

                    # The vectorstore to use to index the child chunks
                    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

                    # The storage layer for the parent documents
                    store = InMemoryStore()
                    id_key = "doc_id"

                    # Custom Retriever to return similarity scores
                    retriever = CustomMultiVectorRetriever(
                        vectorstore=vectorstore,
                        docstore=store,
                        id_key=id_key,
                    )

                    # Add texts
                    if texts:
                        doc_ids = [str(uuid.uuid4()) for _ in texts]
                        summary_texts = [
                            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
                        ]
                        retriever.vectorstore.add_documents(summary_texts)
                        retriever.docstore.mset(list(zip(doc_ids, texts)))

                    # Add tables
                    if tables:
                        table_ids = [str(uuid.uuid4()) for _ in tables]
                        summary_tables = [
                            Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
                        ]
                        retriever.vectorstore.add_documents(summary_tables)
                        retriever.docstore.mset(list(zip(table_ids, tables)))

                    # Add image summaries
                    if images:
                        img_ids = [str(uuid.uuid4()) for _ in images]
                        summary_img = [
                            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
                        ]
                        retriever.vectorstore.add_documents(summary_img)
                        retriever.docstore.mset(list(zip(img_ids, images)))
                    
                    st.session_state.processed_file = file
                    print("\nFile sucessfuly processed, ready for query: ", st.session_state.processed_file)
                    st.session_state.retriever = retriever

                docs = st.session_state.retriever.invoke(user_query)
                
                chain = (
                    {
                        "context": st.session_state.retriever | RunnableLambda(parse_docs),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(build_prompt)
                    | ChatOpenAI(model="gpt-4o-mini")
                    | StrOutputParser()
                )

                with get_openai_callback() as cb:
                    response = chain.invoke(
                        user_query
                    )
                    print(cb,"\n\n")
                
                with st.chat_message("AI"):
                    st.markdown(response)   

                if mime_type(file) == "application/pdf":
                    with st.expander("Possibly Related Sources"):
                        # st.write("Similarity score is between 0 and 1. The higher the score, the more similar the source is to the query.")
                        min_score = 0.7 # after testing, this is the optimal score to filter out unrelated sources
                        filtered_docs = [doc for doc in docs if doc["sub_docs"][0].metadata["score"] > min_score]
                        if filtered_docs:
                            for doc in docs:
                                st.write("Similarity score:", doc["sub_docs"][0].metadata["score"])
                                display_chunk_pages(file, doc['doc'])
                        else:
                            st.write("No related sources found.")
                else:
                    st.write("For more visualisation on related sources, please upload the document as a PDF file.")
                

            except UnsupportedFileFormatError as e:
                st.error(f"Error: File type is not supported. \n\nList of supported file types: https://docs.unstructured.io/platform-api/supported-file-types")

            


if __name__ == "__main__":
    main()