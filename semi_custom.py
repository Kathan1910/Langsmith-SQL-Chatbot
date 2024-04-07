import os 
import re
import time
import pandas as pd 
import streamlit as st 
from langchain.sql_database import SQLDatabase
from openai import OpenAI
from sqlalchemy import create_engine, text
import google.generativeai as genai
from langsmith import wrappers


# Environment variable for langsmith tracing
os.environ['LANGCHAIN_API_KEY'] = "LANGCHAIN_API_KEY"
os.environ['LANGCHAIN_TRACING_V2'] = "True"

#Environment variables
#Openai api key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
#GEMINI api key
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Database connection
db_user = "postgres" #Add your database creditional
db_password = "Password" #Add your database creditional
db_host = "localhost" #Add your database creditional
db_name = "schoolDB" #Add your database creditional
db_port = 5432 #Add your database creditional
SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URL)


#database
db_name = "postgresql"


# Model initialization
#tracing using wrappers
client = wrappers.wrap_openai(OpenAI())     # Wtapping openai for tracing calls in langsmith
#GEMINI Model
gemini_model = genai.GenerativeModel('gemini-pro')

#table information
table_info = """
        -> The database contains the following tables:

    1. Students: Contains information about students
        - StudentID: Unique identifier for the student
        - FirstName: First name of the student
        - LastName: Last name of the student
        - DateOfBirth: Student's birthdate
        - Grade: The grade or class the student is in
        - Email: Student's email address
        - Address: Home address of the student

    2. Teachers: Contains information about teachers
        - TeacherID: Unique identifier for the teacher
        - FirstName: First name of the teacher
        - LastName: Last name of the teacher
        - Email: Teacher's email address
        - Specialty: The subject or area the teacher specializes in
        - PhoneNumber: Teacher's contact number

    3. Courses: Contains information about courses offered
        - CourseID: Unique identifier for the course
        - CourseName: Name of the course
        - Description: A brief description of the course content
        - TeacherID: The identifier for the teacher responsible for the course

    4. Enrollments: Tracks student enrollments in courses
        - EnrollmentID: Unique identifier for the enrollment record
        - StudentID: Identifier for the student
        - CourseID: Identifier for the course
        - EnrollmentDate: The date when the student enrolled in the course
        - Grade: The grade received by the student in the course

    5. Classrooms: Contains information about classrooms
        - ClassroomID: Unique identifier for the classroom
        - ClassroomName: Name or number of the classroom
        - Location: The physical location of the classroom
        - Capacity: How many individuals the classroom can accommodate

    6. Schedule: Organizes when and where courses take place
        - ScheduleID: Unique identifier for the schedule entry
        - CourseID: Identifier for the course
        - ClassroomID: Identifier for the classroom where the course is held
        - StartTime: When the course begins
        - EndTime: When the course ends
        - DaysOfWeek: Which days of the week the course is held on
"""

# Define the prompt to be passed to the model
prompt = f"""
    -> If the input is a greeting, respond appropriately with a greeting message, return only greeting message for any greeting input.\n
    -> Do not provide any query for update and deletion related tasks.
    -> You are an expert in translating English questions into detailed SQL queries for {db_name} database. Your task is to create a robust SQL query based on the user input using given table information.
    -> If the user question is not suitable for a SQL query, then respond appropriately as per the question.
    -> Ensure the use of single quotes ('') throughout the query, excluding double quotes. Use three backticks (```) at the beginning and end of the query, do not use \\n in the query.
    -> always limit the query to retrieve information for only 10 datapoints each time.\n
    -> For date-related operations, use the 'EXTRACT' function.\n
    -> The database contains the following tables; focus on the relevant table for the given information. Use the following information to form the query:\n
    
    {table_info}
"""


#Content of system role for query generator using OpenAI
query_generator_task = f"You are an expert in translating English questions into detailed SQL queries for {db_name} database. Your task is to create a robust {db_name} query based on the given table informations. If the question is not suitable for a SQL query, return none."


#Content of system role for answer generator using OpenAI
answer_generator_task = "You're tasked with generating an expert answer based on the provided data. Your goal is to craft a comprehensive response derived exclusively from the given information. Ensure that your answer is thorough and directly relates to the provided data. If requires then present data in tabular format."


# input Prompt Generator using User Input
def generate_prompt(question, prompt):
    pt = f"""
        **User Question:**
        {question}
        **Additional Information:**
        {prompt}
    """
    return pt


#OpenAI Model initialization
def openai_model(full_query,task):
    messages=[
        {"role": "system",
        "content":task},
        {"role": "user",
        "content":full_query},
        ]
    
    response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            temperature=0,
            max_tokens=1500
        )
    return response



# This function retrieves a SQL query from the model output.
def get_query(output):
    
    # Define a regular expression pattern to find the SQL query enclosed between ```sql and ```
    pattern = re.compile(r"```sql\n(.*?)\n```", re.DOTALL)

    # Search for the pattern in the output string
    match = pattern.search(output)

    # Check if a match is found
    if match:
        # Extract the SQL query
        sql_query = f'{match.group(1).strip()}'
        return sql_query
    else:
        # Inform if no query is found
        print('Query is not found.')


# This function fetches data from the database using a specified SQL query.
def get_data(query):
    # Create a database engine
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    try:
        # Use a context manager to handle connections and transactions automatically
        with engine.connect() as connection:
            # Convert the query into a format suitable for execution
            query = text(query)

            # Execute the query
            result = connection.execute(query)

            # Fetch all the rows returned by the query
            rows = result.fetchall()
            return rows

    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error: {e}")


# This function generates a final prompt for an expert answer generator based on the provided inputs.
def generate_final_prompt(question, sql, data, tables):
    final_prompt_template = f"""
        ==> You are an expert answer generator. Your task is to form a comprehensive answer from the given data.\n
        ==> If the data is not available, provide an answer that the data is not available in your language.\n
        ==> If the user input is greeting then greet the user.\n
        ==> Do not return any SQL query or any irrelevant data.\n
        ==> You do not have permission to Delete or Update any table, if user ask to do it reply approprietly that you do not have permission to update database.\n
        **User Question:**
        {question}

        **SQL Query:**
        {sql}

        **Table information:**
        {tables}

        **Data:**
        {data}

    Note: 
        ==> Your task is to generate an appropriate answer based on the given question using the provided data. Consider the SQL Query as a reference.\n
        ==> All commam seperated data should be presented in tabular format only 
        ==> If you are unable to find a relevant answer, please state that there is insufficient data available.\n

        ### Please craft a well-formed response.
    """
    return final_prompt_template


# This function extracts text content from a response object, specifically designed for GEMINI model output.
def get_text_content(response):
    """
    Extract text content from a response object.
    Parameters:
    - response: The response object containing text content.
    Returns:
    - text_content: Extracted text content.
    """
    if len(response.parts) == 1 and "text" in response.parts[0]:
        # Access the text content if it's a simple text response
        text_content = response.parts[0].text
    else:
        # Handle the case where the response is not a simple text
        # Loop through parts and concatenate text from each part
        text_content = ""
        for part in response.parts:
            if "text" in part:
                text_content += part.text

    return text_content


# This function calculates the cost based on custom tokens for input and completion.

def cost_counter(query_output, answer_output):
    # Initialize variables to store the number of tokens for completion and prompt
    query_completion_tokens = 0
    query_prompt_tokens = 0
    answer_completion_tokens = 0
    answer_prompt_tokens = 0

    # Check if query output is available and assign the number of tokens accordingly
    if query_output:
        query_completion_tokens = query_output.completion_tokens
        query_prompt_tokens = query_output.prompt_tokens
    
    # Check if answer output is available and assign the number of tokens accordingly
    if answer_output:
        answer_completion_tokens = answer_output.completion_tokens
        answer_prompt_tokens = answer_output.prompt_tokens

    # Define the cost per token for input and completion based on model
    input_cost_per_token = 0.0005           #Change this cost as per model cost 
    completion_cost_per_token = 0.0015      #Change this cost as per model cost 

    # Calculate the total number of tokens for input and completion
    total_input_tokens = query_prompt_tokens + answer_prompt_tokens
    total_completion_tokens = query_completion_tokens + answer_completion_tokens

    # Calculate the total cost for input and completion tokens
    total_input_cost = ((total_input_tokens / 1000) * input_cost_per_token)
    total_completion_cost = ((total_completion_tokens / 1000) * completion_cost_per_token)

    # Calculate the total cost by summing input and completion costs
    total_cost = total_input_cost + total_completion_cost

    return total_cost


# Streamlit app
def main():
    st.title("Universal Chatbot: Semi Custom Model")

    # Option for generating final answer in the sidebar
    query_model = st.sidebar.radio("Choose a model to generate SQL Query:", ("Gemini","OpenAI"), index=1)
    print(f"Query model: {query_model}")

    # Option for generating final answer in the sidebar
    answer_model = st.sidebar.radio("Choose a model to generate the final answer:", ("Gemini","OpenAI"), index=0)
    print(f"Answer model: {answer_model}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_query = st.chat_input("Enter your query:")
    print(user_query)
    if user_query:
        full_query = generate_prompt(question=user_query, prompt=prompt)

    if user_query:
        # Displaying the User Message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        #timer Started
        start_time = time.time()

        #SQL Query Generator
        query_token_details = None
        if query_model =='OpenAI':
            model_response = openai_model(full_query=full_query,task=query_generator_task)
            response = model_response.choices[0].message.content
            query_token_details = model_response.usage
            real_query = get_query(response)


        elif query_model == 'Gemini':
            #SQL Query Generator
            model_response = gemini_model.generate_content(full_query)
            response = get_text_content(model_response)
            real_query = get_query(response)

        # Displaying the Assistant Message
        print(f"Query generator response: {response}")

        # #Query Performer
        final_output = get_data(real_query)
        ## Displaying the data retrieved from query
        print(f"data retrieved from query: {final_output}")

        # Final result formatter 
        answer_token_details = None
        if answer_model == 'OpenAI':
            #Final result using openai
            full_prompt = generate_final_prompt(question=user_query,sql=real_query,data=final_output,tables=table_info)
            answer = openai_model(full_query=full_prompt,task=answer_generator_task)
            final_answer = answer.choices[0].message.content
            answer_token_details = answer.usage
            # # Displaying and Storing the Assistant Message
            with st.chat_message("assistant"):
                st.markdown(final_answer)
        elif answer_model == 'Gemini':
            #Final result using gemini
            full_prompt = generate_final_prompt(question=user_query,sql=real_query,data=final_output,tables=table_info)
            answer = gemini_model.generate_content(full_prompt)
            final_answer = get_text_content(answer)
            # # Displaying and Storing the Assistant Message
            with st.chat_message("assistant"):
                st.markdown(final_answer)
            
        #Displaying the final answer
        print(f"Final answer from model: {final_answer}")

        # Timer ended
        end_time = time.time()
        #Total time taken 
        elapsed_time = end_time - start_time

        total_cost = cost_counter(query_output=query_token_details,answer_output=answer_token_details)
        
        #printing data
        details = (f"""
            Query Token:    {query_token_details}\n
            Answer Token:   {answer_token_details}\n
            Total Time:     {elapsed_time}\n
            Total Cost:     {round(total_cost,5)} $\n
        """)
        print(f"Details: {details}")
        print(100*"-")

        # Storing Messages
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()