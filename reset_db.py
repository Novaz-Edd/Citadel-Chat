"""Quick script to reset the database: remove all users and recreate clean."""
from sqlalchemy import create_engine, text

e = create_engine("sqlite:///citadel_users.db")
conn = e.connect()
conn.execute(text("DELETE FROM users"))
conn.execute(text("DELETE FROM messages"))
conn.commit()
print("All users and messages deleted.")
conn.close()
