import asyncio
from DatabaseManager import DatabaseManager  # Ensure this file exists
from datetime import datetime, timedelta

# === CONFIGURE DATABASE CREDENTIALS ===
DB_USER = "ident_tdr"
DB_PASSWORD = "8g76yfg87hs8gf7hfs8hgf"
DB_HOST = "127.0.0.1"
DB_NAME = "Identificator_db"

async def test_database():
    db = DatabaseManager(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

    # Test Data
    test_email = "testuser@example.com"
    test_token = "sample_token_123"
    token_expiry = (datetime.utcnow() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    print("---- TEST: Create User ----")
    success = await db.create_user(test_email, test_token, token_expiry)
    print(f"User creation: {'Success' if success else 'Failed'}")

    print("---- TEST: Retrieve User by Email ----")
    user = await db.get_user_by_email(test_email)
    print(f"User found: {user}" if user else "User not found.")

    print("---- TEST: Retrieve User by Token ----")
    user_by_token = await db.get_user_by_token(test_token)
    print(f"User found by token: {user_by_token}" if user_by_token else "User not found by token.")

    print("---- TEST: Update Token ----")
    new_token = "updated_token_456"
    new_expiry = (datetime.utcnow() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
    update_success = await db.update_user_token(test_email, new_token, new_expiry)
    print(f"Token update: {'Success' if update_success else 'Failed'}")

    print("---- TEST: Log User Login ----")
    user_id = user['id'] if user else 1  # Use retrieved user ID or fallback
    ip_address = "192.168.1.100"
    user_agent = "Test Browser v1.0"
    login_success = await db.log_user_login(user_id, ip_address, user_agent, success=True)
    print(f"Login logged: {'Success' if login_success else 'Failed'}")

    print("---- TEST COMPLETE ----")

if __name__ == "__main__":
    asyncio.run(test_database())  # <-- Run the async function properly
