
import psycopg2
import pandas as pd

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

    # Create `job_questions` table with proper indexing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_questions (
            id SERIAL PRIMARY KEY, 
            job_role VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            difficulty VARCHAR(50),
            category VARCHAR(255),
            source_api VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

    # Function to insert CSV data into PostgreSQL (only first 50 questions)
    def insert_questions(csv_file, job_role, category, source):
        try:
            # Read CSV
            df = pd.read_csv(csv_file, on_bad_lines="skip", encoding="utf-8")

            # Check if 'Question' column exists
            if "Question" not in df.columns:
                print(f"Error: Column 'Question' not found in {csv_file}. Available columns: {df.columns}")
                return

            # Select first 50 questions
            df = df.head(50)

            # Convert DataFrame into a list of tuples for batch insert
            question_data = [(job_role, row["Question"], "Medium", category, source) for _, row in df.iterrows()]

            # Batch insert using `executemany` (faster than looping)
            cursor.executemany("""
                INSERT INTO job_questions (job_role, question, difficulty, category, source_api)
                VALUES (%s, %s, %s, %s, %s)
            """, question_data)

            conn.commit()
            print(f"Stored {len(df)} questions from {csv_file} into job_questions table.")
        except Exception as e:
            print(f"Unexpected Error with {csv_file}: {e}")

    # Insert first 50 Data Science Interview Questions
    insert_questions(r"C:\Users\ASUS\Downloads\data science .csv", "Data Scientist", "technical", "CSV")

except psycopg2.Error as e:
    print(f"Database connection error: {e}")

finally:
    # Ensure the connection is properly closed
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()

print("First 50 Data Science questions stored successfully!")
