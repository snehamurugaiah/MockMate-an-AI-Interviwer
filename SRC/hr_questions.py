
import psycopg2
import requests

# PostgreSQL Connection
try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="dhivya@2005",  # Replace with actual password
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Ensure the HR_questions table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS HR_questions (
            id SERIAL PRIMARY KEY,
            job_role VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            difficulty VARCHAR(50),
            source_api VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

    # Open Trivia API URL
    url = "https://opentdb.com/api.php?amount=10&type=multiple"

    # Fetch data from API
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the request fails
    data = response.json()

    # Prepare data for batch insert
    question_data = [
        (
            question["category"],  # Using category as job role
            question["question"],
            question["difficulty"].capitalize(),  # Capitalize difficulty
            "Open Trivia API"
        )
        for question in data.get("results", [])
    ]

    # Batch insert if there are questions
    if question_data:
        cursor.executemany("""
            INSERT INTO HR_questions (job_role, question, difficulty, source_api)
            VALUES (%s, %s, %s, %s)
        """, question_data)
        conn.commit()
        print(f"✅ Stored {len(question_data)} questions successfully in HR_questions!")

except requests.exceptions.RequestException as e:
    print(f"❌ API request error: {e}")
except psycopg2.Error as e:
    print(f"❌ Database error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    # Ensure connection is properly closed
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
