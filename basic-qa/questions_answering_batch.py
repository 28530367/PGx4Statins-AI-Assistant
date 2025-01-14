# -*- coding:utf-8 -*-
# Created by liwenw at 7/18/23

# python3 questions_answering_batch.py -y /media/disk2/HSW/PGx4Statins-AI-Assistant/config.yaml -r patient

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os
from chromadb.config import Settings
import chromadb

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from omegaconf import OmegaConf
import argparse
from templates import system_provider_template, human_provider_template, system_patient_template, human_patient_template, system_HSW_template, human_HSW_template
from patient_questions import patient_questions
from provider_questions import provider_questions
from HSW_questions import HSW_questions

def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to question/answer.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    parser.add_argument("-r", "--role", dest="role",
                        help="role(patient/provider/HSW) for question/answering", metavar="ROLE")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.yamlfile is None:
        parser.print_help()
        exit()

    yamlfile = args.yamlfile
    config = OmegaConf.load(yamlfile)

    model = ChatOllama(
        model=config.ollama.chat_model_name,
        temperature=0.0,
    )

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    collection_name = config.chromadb.collection_name
    persist_directory = config.chromadb.persist_directory

    persistent_client = chromadb.PersistentClient(path=persist_directory)

    vector_store = Chroma(collection_name=collection_name,
                          client=persistent_client,
                          embedding_function=embeddings,
                          )

    chat_search_type = config.ollama.chat_search_type
    chat_search_k = config.ollama.chat_search_k
    retriever = vector_store.as_retriever(search_type=chat_search_type, search_kwargs={"k": chat_search_k})

    if args.role == "provider":
        questions = provider_questions
        system_template = system_provider_template
        human_template = human_provider_template
    elif args.role == "patient":
        questions = patient_questions
        system_template = system_patient_template
        human_template = human_patient_template
    elif args.role == "HSW":
        questions = HSW_questions
        system_template = system_HSW_template
        human_template = human_HSW_template
    else:
        print("role not supported")
        exit()

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=8192,
        chain_type_kwargs=chain_type_kwargs,
        verbose=False,
    )

    # 紀錄
    log_str = ""
    k = 0
    for question in questions:
        k += 1
        print(f"question: {question}")
        log_str += f"## {k}. question:" + "\n" 
        log_str += question + "\n" 
        # Get answer
        response = chain.invoke(question)
        answer = response["answer"]
        source = response["source_documents"]

        # Display answer
        print("\nSources:")
        log_str += f"## {k}. Sources:" + "\n"
        for i, document in enumerate(source, start=1):
            print(f"ID[{i}]: {document}")
            log_str += f"ID[{i}]: {document}" + "\n"
        print(f"\nAnswer: {answer}")
        print("--------------------------------------------------")

        log_str += f"## {k}. Answer:" + "\n"
        log_str += answer + "\n"
        log_str += f"--------------------------------------------------" + "\n"

    file_path = f"/media/disk2/HSW/PGx4Statins-AI-Assistant/batch_output/{args.role}_batch3.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(log_str)
    print(f"訓練紀錄已寫入 {file_path}")

if __name__ == "__main__":
    main()
