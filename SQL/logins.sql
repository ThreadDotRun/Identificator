CREATE TABLE logins (
    id INT AUTO_INCREMENT PRIMARY KEY,               -- Unique login record ID
    user_id INT NOT NULL,                            -- References users.id
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Timestamp of login
    ip_address VARCHAR(45) NOT NULL,                 -- User's IP address (IPv4/IPv6)
    user_agent TEXT DEFAULT NULL,                    -- Browser/User-Agent info
    success BOOLEAN DEFAULT TRUE,                    -- Was the login successful?
    failure_reason TEXT DEFAULT NULL,                -- Stores reason if failed (e.g., "Invalid password")
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_user_id ON logins(user_id);
CREATE INDEX idx_login_time ON logins(login_time);
CREATE INDEX idx_ip_address ON logins(ip_address);
