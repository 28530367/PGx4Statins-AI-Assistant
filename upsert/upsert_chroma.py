# -*- coding:utf-8 -*-
# Created by liwenw at 6/12/23

# python upsert_chroma.py -y ../config.yaml

import os
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb

from omegaconf import OmegaConf
import argparse
from source_metadata_mapping import pick_metadata

def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to chat.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    return parser


def upsert_csv(collection, filename, data_dir, i):
    metadata = pick_metadata(filename)
    # print("metadata", metadata)
    csv_loader = CSVLoader(os.path.join(data_dir, filename))
    pages = csv_loader.load_and_split()
    # print(pages)
    for page in pages:
        # print("page.metadata", page.metadata)
        collection.upsert(
            documents=page.page_content,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i


def upsert_txt(collection, filename, data_dir, i):
    metadata = pick_metadata(filename)
    # print(metadata)
    file = open(os.path.join(data_dir, filename), 'r')
    lines = file.readlines()
    for line in lines:
        collection.upsert(
            documents=line,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i


def upsert_pdf(collection, filename, data_dir, i, chunk_size, chunk_overlap):
    metadata = pick_metadata(filename)
    # print(metadata)
    pdf_loader = PyPDFLoader(os.path.join(data_dir, filename))
    data = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(data)
    for text in texts:
        collection.upsert(
            documents=text.page_content,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.yamlfile is None:
        parser.print_help()
        exit()

    yamlfile = args.yamlfile
    config = OmegaConf.load(yamlfile)
    data_dirs = config.data.directory
    chunk_size = config.parse_pdf.chunk_size
    chunk_overlap = config.parse_pdf.chunk_overlap

    # Create a new Chroma client with persistence enabled.
    persist_directory = config.chromadb.persist_directory
    chroma_client =  chromadb.PersistentClient(path=persist_directory)

    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.ollama.embedding_model_name)

    collection_name = config.chromadb.collection_name
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=chroma_ef)
    i=collection.count()

    for data_dir in data_dirs:
        print(f"Ingest files at {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf") and os.path.isfile(os.path.join(data_dir, filename)):
                print(f"Upserting {filename}")
                i = upsert_pdf(collection, filename, data_dir, i, chunk_size, chunk_overlap)
            elif filename.endswith(".csv") and os.path.isfile(os.path.join(data_dir, filename)):
                print(f"Upserting {filename}")
                i = upsert_csv(collection, filename, data_dir, i)
            elif filename.endswith(".txt") and os.path.isfile(os.path.join(data_dir, filename)):
                print(f"Upserting {filename}")
                i = upsert_txt(collection, filename, data_dir, i)
            else:
                continue


if __name__ == "__main__":
    main()