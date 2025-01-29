import asyncmy
from asyncmy import connect
import asyncio

class DatabaseManager:
    def __init__(self, db_user, db_password, db_host, db_name):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_name = db_name

    async def _get_connection(self):
        return await connect(
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            database=self.db_name
        )

    # Example: Create User
    async def create_user(self, email, token, token_expiry):
        conn = await self._get_connection()
        try:
            async with conn.cursor() as cursor:
                query = """
                INSERT INTO users (email, token, token_expiry)
                VALUES (%s, %s, %s)
                """
                await cursor.execute(query, (email, token, token_expiry))
                await conn.commit()
                return True
        except Exception as e:
            print(f"Error in creating user: {e}")
            return False
        finally:
            await conn.close()

    # Example: Retrieve User by Email
    async def get_user_by_email(self, email):
        conn = await self._get_connection()
        try:
            async with conn.cursor() as cursor:
                query = "SELECT * FROM users WHERE email = %s"
                await cursor.execute(query, (email,))
                result = await cursor.fetchone()
                return result
        except Exception as e:
            print(f"Error in getting user: {e}")
            return None
        finally:
            await conn.close()

    # Other methods like update_user_token, log_user_login, etc. would follow the same pattern.
