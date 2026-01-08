import sqlite3
import uuid

DB_FILE = "orders.db"


def initialize_db():
    """Initialize SQLite database and create table if it doesn't exist"""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            item TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_order(username: str, item: str, quantity: int) -> str:
    """Create a new order and save it to database. Returns (order_id, message)"""
    initialize_db()
    order_id = str(uuid.uuid4())[:8]
    status = "Utworzone"
    
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO orders (order_id, username, item, quantity, status) VALUES (?, ?, ?, ?, ?)",
        (order_id, username, item, quantity, status)
    )
    conn.commit()
    conn.close()
    
    message = f"Zamówienie {order_id} zostało utworzone dla {item} w ilości {quantity}"
    return message


def get_order_status(order_id: str, username: str) -> str:
    """Check order status from database. Returns status message"""
    initialize_db()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("SELECT status FROM orders WHERE order_id = ? AND username = ?", (order_id, username))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return f"Status zamówienia {order_id} to: {row[0]}"
    
    return f"Nie znaleziono zamówienia {order_id} dla użytkownika {username}"


def cancel_order(order_id: str, username: str, reason: str) -> str:
    """Cancel an order and update its status in database. Returns status message"""
    initialize_db()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("UPDATE orders SET status = ? WHERE order_id = ? AND username = ?", ("Anulowane", order_id, username))
    conn.commit()
    conn.close()
    
    if cursor.rowcount == 0:
        return f"Nie znaleziono zamówienia {order_id} dla użytkownika {username}"
    
    return f"Zamówienie {order_id} zostało anulowane z powodu: {reason}"
