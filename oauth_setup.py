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
    'tweet.read',
    'tweet.write',
    'users.read',
    'follows.read',
    'follows.write',
    'like.read',
    'like.write',
    'offline.access'
]

def generate_code_verifier() -> str:
    """Generate a code verifier for PKCE"""
    code_verifier = secrets.token_urlsafe(32)
    return code_verifier

def generate_code_challenge(verifier: str) -> str:
    """Generate a code challenge for PKCE"""
    code_challenge = hashlib.sha256(verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode('utf-8').replace('=', '')
    return code_challenge

@app.route('/')
def home():
    """Start OAuth flow"""
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
    auth_url = (
        'https://twitter.com/i/oauth2/authorize'
        f'?response_type=code'
        f'&client_id={CLIENT_ID}'
        f'&redirect_uri={REDIRECT_URI}'
        f'&scope={scope}'
        f'&state={state}'
        '&code_challenge_method=S256'
        f'&code_challenge={code_challenge}'
    )
    
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
        
        try:
            # Try to add token_type column if it doesn't exist
            c.execute('ALTER TABLE oauth_tokens ADD COLUMN token_type TEXT')
        except sqlite3.OperationalError:
            # Column already exists, ignore the error
            pass
            
        # Calculate expiration time
        expires_at = (datetime.now() + timedelta(seconds=token_data['expires_in'])).isoformat()
        
        try:
            # Try to insert with token_type
            c.execute('''
                INSERT OR REPLACE INTO oauth_tokens 
                (id, access_token, refresh_token, expires_at, token_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                1, 
                token_data['access_token'],
                token_data['refresh_token'],
                expires_at,
                token_data.get('token_type', 'Bearer')
            ))
        except sqlite3.OperationalError:
            # Fall back to old schema if token_type column doesn't exist
            c.execute('''
                INSERT OR REPLACE INTO oauth_tokens 
                (id, access_token, refresh_token, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (
                1, 
                token_data['access_token'],
                token_data['refresh_token'],
                expires_at
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