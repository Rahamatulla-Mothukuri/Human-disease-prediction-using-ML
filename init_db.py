import sqlite3
import os

# Get path to current directory (where the script is located)
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, 'users.db')

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop old table if exists
cursor.execute("DROP TABLE IF EXISTS users")

# Create new users table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    mobile TEXT,
    age INTEGER,
    address TEXT
)
''')

conn.commit()
conn.close()
print(f"✅ Database created at {db_path} and users table initialized.")
