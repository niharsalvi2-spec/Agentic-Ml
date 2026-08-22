# Database Collection

Most real-world company data lives here — this is the most common production source.

## SQLAlchemy — works with any backend

```python
from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL
engine = create_engine("postgresql://username:password@localhost:5432/dbname")
# MySQL
engine = create_engine("mysql+pymysql://username:password@localhost/dbname")
# SQLite (local file)
engine = create_engine("sqlite:///mydata.db")

query = """
    SELECT customer_id, age, salary, purchase_amount
    FROM customers
    WHERE age > 25
    ORDER BY salary DESC
    LIMIT 1000
"""
df = pd.read_sql(query, engine)
```

Never build queries with f-strings/string concatenation from user input — use bound parameters:

```python
query = "SELECT * FROM customers WHERE age > %(min_age)s"
df = pd.read_sql(query, engine, params={"min_age": 25})
```

## psycopg2 — direct PostgreSQL

```python
import psycopg2
import pandas as pd

conn = psycopg2.connect(host="localhost", database="mydb", user="postgres", password="yourpassword")
cursor = conn.cursor()
cursor.execute("SELECT name, age, salary FROM employees WHERE department = %s", ("Engineering",))

rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]
df = pd.DataFrame(rows, columns=columns)

cursor.close()
conn.close()
```

## Core SQL patterns for extraction

```sql
-- Basic extraction
SELECT * FROM table_name LIMIT 100;

-- Filter
SELECT * FROM orders WHERE amount > 1000;

-- Join
SELECT c.name, o.amount, o.date
FROM customers c
JOIN orders o ON c.id = o.customer_id;

-- Aggregate
SELECT department, AVG(salary), COUNT(*)
FROM employees
GROUP BY department;

-- Date filter
SELECT * FROM logs
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
```

## Notes

- Always close connections (or use a `with` / context manager) — leaked connections are the most common bug in collector scripts.
- For large extractions, page through with `LIMIT`/`OFFSET` or a keyset cursor rather than loading an entire table into memory at once.
- Credentials go in environment variables or a secrets manager, never in the script.
