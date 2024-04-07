
# Universal SQL Chatbot Applications 🚀

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integrated-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![LangSmith](https://img.shields.io/badge/LangSmith-Integrated-orange)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Detailed Script Overview](#detailed-script-overview)
- [Hypothetical Examples](#hypothetical-examples)
- [Troubleshooting](#troubleshooting)
- [Documentation Links](#documentation-links)
- [Contributors](#contributors)
- [License](#license)

## Introduction 📖
This project comprises a suite of Streamlit chatbot applications designed to facilitate seamless interaction between users and SQL databases. By leveraging advanced AI models through LangChain and detailed tracing with Langsmith, these applications offer an intuitive conversational interface for querying databases. Targeted at developers, data analysts, and individuals seeking to integrate natural language processing capabilities with database management, this project highlights the power of combining OpenAI's GPT models and Google's Gemini model with robust database interaction capabilities.

## Installation 💾

### Prerequisites
- Python 3.9+
- Streamlit
- pandas
- SQLAlchemy
- LangChain
- Langsmith
- google-generativeai 

### Setup Instructions
1. Clone the repository:
   ```sh
   git clone <repository-url>
   ```
2. Install the required Python packages:
   
   ```sh
   pip install streamlit pandas sqlalchemy langchain langsmith google-generativeai
   ```
3. Configure the necessary environment variables for LangChain API, OpenAI API, and Google API keys as described in the configuration section.

## Usage 🖥️
- To run any of the chatbot applications, navigate to the project directory and use the following command, replacing `<script_name>.py` with the name of the script you wish to execute:
```sh
streamlit run <script_name>.py
```

## Features ✨
- **Multi-Model Support**: Integrates OpenAI's GPT and Google's Gemini models to process natural language queries and generate SQL commands.
- **LangChain Integration**: Utilizes LangChain for orchestrating interactions between the conversational AI and the database, enhancing query accuracy and response relevance.
- **Langsmith Tracing**: Employs Langsmith for detailed tracing of LLM processing, ensuring comprehensive monitoring and analytics of model interactions.
- **Database Interactivity**: Offers direct interaction with SQL databases, supporting a variety of queries from simple data retrieval to complex aggregations.
- **Customizable Query Processing**: Allows for the customization of query processing and response generation, tailored to specific database schemas and user needs.

## Dependencies 🛠️
- Streamlit: For creating the web application interface.
- pandas: For data manipulation and analysis.
- SQLAlchemy: For database interaction.
- LangChain: For managing LLM operations and database toolkit integration.
- Langsmith: For tracing and monitoring LLM processing.
- google-generativeai: For integrating Google's Gemini model.

## Configuration ⚙️
Set the following environment variables according to your setup:

- `LANGCHAIN_API_KEY`: Your LangChain API key.
- `OPENAI_API_KEY`: Your OpenAI API key for GPT models.
- `GOOGLE_API_KEY`: Your API key for Google's Gemini model.
- Database credentials and connection strings as per each script's requirements.

## Detailed Script Overview 📜
### `custom_tool.py`
- Integrates LangChain with custom greeting tool functionality, demonstrating the use of Langsmith for tracing model interactions and providing an interface for SQL database queries with emphasis on greeting inputs.

### `sql_app.py`
- Showcases a straightforward application of LangChain and OpenAI's GPT model for generating SQL queries, with detailed token usage and response timing logged by Langsmith.

### `semi_custom.py`
- A complex example combining OpenAI and Google's Gemini model for query generation and response formatting, highlighting cost calculations and detailed monitoring with Langsmith.

### `postgres.py` & `gemini.py`
- Illustrate the use of OpenAI and Google's Gemini models, respectively, with PostgreSQL and MySQL databases, showcasing the flexibility of the chatbot applications in handling different database systems and emphasizing the Langsmith tracing capabilities.

## Hypothetical Examples 🌟
- **User Query**: "Show me the top 10 students by performance."
- **Expected SQL Command**: "SELECT * FROM students ORDER BY performance DESC LIMIT 10;"
- **Langsmith Trace Output**: Detailed trace logs showing the LLM's processing steps, token usage, and response time.

## Troubleshooting 🔧
- Ensure all environment variables are correctly set.
- Verify database connection details are accurate and the database server is accessible.
- Check for any API rate limits or restrictions on the OpenAI and Google API keys.

## Documentation Links 📚
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://docs.langchain.com)
- [Langsmith Documentation](https://www.langchain.com/langsmith)
- [OpenAI API Documentation](https://beta.openai.com/docs/)

## Contributors 👥
- To contribute to this project, please fork the repository and submit a pull request.

## License 📝
- This project is licensed under the MIT License - see the LICENSE file for details.
