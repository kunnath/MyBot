# Streamlit LangChain App

This repository contains a Streamlit application that uses LangChain and HuggingFace models to load, split, and analyze PDF documents.

## Prerequisites

- Docker installed on your machine

## Build Docker Image

To build the Docker image, run the following command:

```sh
docker build -t streamlit-langchain-app .


docker run -p 8501:8501 streamlit-langchain-app

