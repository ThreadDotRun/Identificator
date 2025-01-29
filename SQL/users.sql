CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,                -- Unique user ID
    email VARCHAR(255) NOT NULL UNIQUE,               -- User email (acts as username)
    password_hash VARCHAR(255) NOT NULL,              -- Hashed password for security
    auth_token VARCHAR(255) DEFAULT NULL,             -- Current authentication token
    token_expiry DATETIME DEFAULT NULL,               -- Expiry time for the token
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- Account creation timestamp
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Last update
    last_login TIMESTAMP NULL DEFAULT NULL,           -- Last successful login timestamp
    is_active BOOLEAN DEFAULT TRUE                    -- Whether the account is active
);

-- Indexes for fast lookup
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_auth_token ON users(auth_token);
