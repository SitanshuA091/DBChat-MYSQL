# DBChat

A simple **Streamlit** application that allows users to interact with a **MySQL database** using natural language queries. The app generates SQL queries and a natural language response based on user input and returns both the SQL result and a natural language explanation.

---
![Python](https://img.shields.io/badge/Python-3.10-blue)

## Features

- Connect to a MySQL database from the sidebar.
- Ask questions in plain English and get SQL queries executed.
- Receive a natural language explanation of the SQL results.
- Maintains a chat history during the session.
- Uses **GPT OSS 120B** via **Groq API** for query generation and response explanation.

## Usage
- clone the repo `git clone https://github.com/SitanshuA091/DBChat-MYSQL`
- replace the empty values for keys .env .example file with API key obtained from `https://console.groq.com/keys` and rename to .env
- install the requirements using
  ```bash
  pip install -r requirements.txt
  ```
- run the application using
  ```bash
  streamlit run app.py
  ```
- connect to your local mysql database by providing password and other credentials (database name..etc.)



