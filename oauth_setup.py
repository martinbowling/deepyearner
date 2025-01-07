"""OAuth setup for Twitter API v2"""
from flask import Flask, request, redirect, session
import os
import requests
import sqlite3
import base64
import hashlib
import secrets
import urllib.parse
from datetime import datetime, timedelta
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Load environment variables
load_dotenv()

# OAuth 2.0 settings
CLIENT_ID = os.getenv('OAUTH_CLIENT_ID')
CLIENT_SECRET = os.getenv('OAUTH_CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:5000/oauth/callback'

# Define scopes individually for clarity
SCOPES = [
    "tweet.read",
    "tweet.write",
    "users.read",
    "follows.read",
    "follows.write",
    "offline.access",
    "list.read",
    "list.write"
]

def init_db():
    """Initialize database with oauth_tokens table"""
    conn = sqlite3.connect('bot.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            id INTEGER PRIMARY KEY,
            access_token TEXT,
            refresh_token TEXT,
            expires_at TEXT,
            token_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def generate_code_verifier() -> str:
    """Generate a code verifier for PKCE"""
    code_verifier = secrets.token_urlsafe(32)
    return code_verifier

def generate_code_challenge(verifier: str) -> str:
    """Generate a code challenge for PKCE"""
    code_challenge = hashlib.sha256(verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode('utf-8').replace('=', '')
    return code_challenge

def init_oauth_tokens():
    """Initialize OAuth tokens in the database"""
    load_dotenv()
    
    # Get credentials from environment
    client_id = os.getenv('OAUTH_CLIENT_ID')
    client_secret = os.getenv('OAUTH_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise Exception("Missing OAuth credentials in environment")
    
    # Try user authentication first
    auth = (client_id, client_secret)
    response = requests.post(
        'https://api.twitter.com/2/oauth2/token',
        auth=auth,
        headers={
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
    )
    
    if response.status_code != 200:
        print(f"Error getting access token: {response.text}")
        print("Starting OAuth web flow for user authentication...")
        app.run(debug=True, port=5000)
        return
    
    token_data = response.json()
    
    # Store tokens in database
    conn = sqlite3.connect('bot.db')
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            id INTEGER PRIMARY KEY,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TEXT NOT NULL,
            token_type TEXT NOT NULL
        )
    ''')
    
    # Insert or update tokens
    c.execute('''
        INSERT OR REPLACE INTO oauth_tokens (id, access_token, refresh_token, expires_at, token_type)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        1,
        token_data['access_token'],
        token_data.get('refresh_token'),
        (datetime.now() + timedelta(seconds=token_data['expires_in'])).isoformat(),
        token_data['token_type']
    ))
    
    conn.commit()
    conn.close()
    
    print("OAuth tokens initialized successfully")

def get_authorization_url(client_id: str, redirect_uri: str, code_challenge: str, state: str) -> str:
    """Get the authorization URL for OAuth 2.0"""
    scope = " ".join(SCOPES)
    return (
        "https://twitter.com/i/oauth2/authorize"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope}"
        f"&state={state}"
        f"&code_challenge={code_challenge}"
        "&code_challenge_method=S256"
    )

@app.route('/')
def home():
    """Start OAuth flow"""
    # Initialize database
    init_db()
    
    # Generate PKCE codes
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    
    # Generate state
    state = secrets.token_urlsafe(16)
    
    # Store code verifier and state in session
    session['code_verifier'] = code_verifier
    session['oauth_state'] = state
    
    # URL encode scopes
    scope = urllib.parse.quote(' '.join(SCOPES))
    
    # Print loaded credentials
    print("\nLoaded OAuth credentials:")
    print(f"Client ID: {CLIENT_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print(f"Code Verifier: {code_verifier}")
    print(f"Code Challenge: {code_challenge}")
    print(f"State: {state}")
    
    # Generate authorization URL
    auth_url = get_authorization_url(CLIENT_ID, REDIRECT_URI, code_challenge, state)
    
    return redirect(auth_url)

@app.route('/oauth/callback')
def callback():
    """Handle OAuth callback"""
    error = request.args.get('error')
    if error:
        return f"Error during authorization: {error}"
    
    code = request.args.get('code')
    state = request.args.get('state')
    
    if not code:
        return "No code received from Twitter"
    
    # Verify state
    if not state or state != session.get('oauth_state'):
        return "Invalid state parameter"
    
    # Get stored code verifier
    code_verifier = session.get('code_verifier')
    if not code_verifier:
        return "No code verifier found"
        
    # Exchange code for tokens
    token_url = 'https://api.twitter.com/2/oauth2/token'
    
    data = {
        'code': code,
        'grant_type': 'authorization_code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'code_verifier': code_verifier
    }
    
    # Basic auth with client ID and secret
    auth = (CLIENT_ID, CLIENT_SECRET)
    
    response = requests.post(
        token_url,
        auth=auth,
        data=data,
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
        }
    )
    
    if response.status_code == 200:
        token_data = response.json()
        
        # Store tokens in database
        conn = sqlite3.connect('bot.db')
        c = conn.cursor()
        
        # Calculate expiration time
        expires_at = (datetime.now() + timedelta(seconds=token_data['expires_in'])).isoformat()
        
        c.execute('''
            INSERT OR REPLACE INTO oauth_tokens 
            (id, access_token, refresh_token, expires_at, token_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            1, 
            token_data['access_token'],
            token_data['refresh_token'],
            expires_at,
            token_data.get('token_type', 'bearer')
        ))
        
        conn.commit()
        conn.close()
        
        # Clear session
        session.clear()
        
        return "Successfully authenticated! You can close this window."
    else:
        error_data = response.json() if response.text else {'error': 'Unknown error'}
        return f"Error getting token: {error_data}"

if __name__ == '__main__':
    # Allow OAuth without HTTPS locally
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(debug=True, port=5000) 