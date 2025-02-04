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
		try:
			conn = await connect(
				user=self.db_user,
				password=self.db_password,
				host=self.db_host,
				database=self.db_name
			)
			return conn
		except Exception as e:
			print(f"Error connecting to the database: {e}")
			raise  # Re-raise the exception to handle it in the calling method

	async def create_user(self, email, auth_token):
		conn = None
		try:
			conn = await self._get_connection()
			if conn is None:
				print("Failed to establish database connection")
				return False
			
			# Calculate the token expiry time to be 12 hours from now
			token_expiry = datetime.now() + timedelta(hours=12)
			
			async with conn.cursor() as cursor:
				query = "INSERT INTO users (email, auth_token, token_expiry) VALUES (%s, %s, %s)"
				await cursor.execute(query, (email, auth_token, token_expiry))
			await conn.commit()
			print("User created successfully")
			return True
		except Exception as e:
			print(f"Error in creating user: {e}")
			return False
		finally:
			if conn:
				try:
					await conn.close()
					print("Database connection closed")
				except Exception as e:
					print(f"Error closing connection: {e}")



	async def get_user_by_email(self, email):
		conn = None
		try:
			conn = await self._get_connection()
			if conn is None:
				print("Failed to establish database connection")
				return None
			
			async with conn.cursor() as cursor:
				query = "SELECT * FROM users WHERE email = %s"
				await cursor.execute(query, (email,))
				result = await cursor.fetchone()
				if result:
					print(f"User found for email: {email}")
				else:
					print(f"No user found for email: {email}")
				return result
		except Exception as e:
			print(f"Error in getting user: {e}")
			return None
		finally:
			if conn:
				try:
					await conn.close()
					print("Database connection closed")
				except Exception as e:
					print(f"Error closing connection: {e}")

