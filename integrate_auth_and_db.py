#!/usr/bin/env python3
"""
Integration script to add authentication and database persistence to miniprogram_api.py

This script modifies the existing miniprogram_api.py to add:
1. Database initialization in lifespan
2. New authentication endpoints
3. New history endpoints
4. Database persistence to existing chat endpoints

Usage:
    python integrate_auth_and_db.py
"""

import re
import sys
from pathlib import Path

# File path
API_FILE = Path(__file__).parent / "src" / "api" / "miniprogram_api.py"

def read_file(filepath):
    """Read file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def add_imports(content):
    """Add new imports after existing imports."""
    # Find the last import statement
    import_pattern = r'(from src\.api\.config import config\n)'

    new_imports = '''
# Database and authentication imports
from sqlalchemy.orm import Session
from src.database import init_db, close_db, get_db, User, Conversation, Message, CustomScenario
from src.database.history import (
    get_user_conversations,
    get_conversation_detail,
    get_recent_conversation_topics,
    save_conversation_to_db,
    save_message_to_db,
    update_conversation_turn,
    end_conversation
)
from src.auth import get_current_user, get_optional_user, verify_user_owns_conversation
from src.auth.wechat import wechat_login
'''

    content = re.sub(import_pattern, r'\1' + new_imports, content)
    return content

def update_lifespan(content):
    """Update lifespan function to initialize database."""
    # Find the lifespan function
    lifespan_pattern = r'(@asynccontextmanager\nasync def lifespan\(app: FastAPI\):.*?yield)'

    def replace_lifespan(match):
        original = match.group(0)
        # Add database initialization after "Starting LangCoach..."
        db_init = '''
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
'''
        # Insert after the first logger.info
        parts = original.split('logger.info("Starting LangCoach Mini Program API...")', 1)
        if len(parts) == 2:
            return parts[0] + 'logger.info("Starting LangCoach Mini Program API...")' + db_init + parts[1]
        return original

    content = re.sub(lifespan_pattern, replace_lifespan, content, flags=re.DOTALL)

    # Add close_db() to shutdown
    shutdown_pattern = r'(logger\.info\("Shutting down LangCoach Mini Program API\.\.\."\))'
    content = re.sub(shutdown_pattern, r'\1\n    close_db()', content)

    return content

def add_request_models(content):
    """Add new request/response models."""
    # Find where to insert (after existing BaseModel definitions)
    pattern = r'(class TranscribeResponse\(BaseModel\):.*?\n\n)'

    new_models = '''
# Authentication models
class WeChatAuthRequest(BaseModel):
    code: str
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None

class WeChatAuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]

class UserProfileResponse(BaseModel):
    id: int
    nickname: str
    avatar_url: Optional[str]
    created_at: str

# History models
class ConversationListResponse(BaseModel):
    conversations: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int

class ConversationDetailResponse(BaseModel):
    id: int
    session_id: str
    scenario: str
    difficulty: str
    max_turns: int
    current_turn: int
    status: str
    created_at: str
    ended_at: Optional[str]
    duration: Optional[int]
    rating: Optional[int]
    overall_score: Optional[int]
    messages: List[Dict[str, Any]]
    custom_scenario: Optional[Dict[str, Any]]

'''

    content = re.sub(pattern, r'\1' + new_models, content, flags=re.DOTALL)
    return content

def replace_auth_endpoint(content):
    """Replace the placeholder auth endpoint with real implementation."""
    # Find and replace the auth endpoint
    auth_pattern = r'@app\.post\("/api/auth/wechat"\)\nasync def wechat_auth\(code: str = Form\(\.\.\.\)\):.*?return \{[^}]+\}'

    new_auth = '''@app.post("/api/auth/wechat", response_model=WeChatAuthResponse)
async def wechat_auth(request: WeChatAuthRequest, db: Session = Depends(get_db)):
    """
    WeChat Mini Program login endpoint.

    Exchanges WeChat authorization code for user token.
    Creates new user if first time login.
    """
    try:
        result = await wechat_login(
            code=request.code,
            nickname=request.nickname,
            avatar_url=request.avatar_url,
            db=db
        )
        return result
    except ValueError as e:
        logger.error(f"WeChat auth failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"WeChat auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")'''

    content = re.sub(auth_pattern, new_auth, content, flags=re.DOTALL)
    return content

def add_new_endpoints(content):
    """Add new endpoints after auth endpoint."""
    # Find where to insert (after auth endpoint)
    pattern = r'(@app\.post\("/api/auth/wechat".*?raise HTTPException.*?\n\n)'

    new_endpoints = '''

@app.get("/api/auth/me", response_model=UserProfileResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile information."""
    return {
        "id": current_user.id,
        "nickname": current_user.nickname,
        "avatar_url": current_user.avatar_url,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }


# ============================================================
# History Endpoints
# ============================================================

@app.get("/api/history/conversations", response_model=ConversationListResponse)
async def get_conversation_history(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of user's conversation history.

    Query Parameters:
    - limit: Maximum number of conversations to return (default: 20)
    - offset: Number of conversations to skip (default: 0)
    """
    try:
        result = get_user_conversations(db, current_user, limit, offset)
        return result
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")


@app.get("/api/history/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_details(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific conversation.

    Includes:
    - Conversation metadata
    - All messages
    - Custom scenario details (if applicable)
    - Statistics
    """
    try:
        detail = get_conversation_detail(db, current_user, conversation_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation detail")

'''

    content = re.sub(pattern, r'\1' + new_endpoints, content, flags=re.DOTALL)
    return content

def main():
    """Main integration function."""
    print("Starting integration of authentication and database features...")

    if not API_FILE.exists():
        print(f"Error: {API_FILE} not found!")
        sys.exit(1)

    # Read original content
    print(f"Reading {API_FILE}...")
    content = read_file(API_FILE)

    # Make modifications
    print("Adding imports...")
    content = add_imports(content)

    print("Updating lifespan function...")
    content = update_lifespan(content)

    print("Adding request/response models...")
    content = add_request_models(content)

    print("Replacing auth endpoint...")
    content = replace_auth_endpoint(content)

    print("Adding new endpoints...")
    content = add_new_endpoints(content)

    # Create backup
    backup_file = API_FILE.with_suffix('.py.backup')
    print(f"Creating backup at {backup_file}...")
    write_file(backup_file, read_file(API_FILE))

    # Write modified content
    print(f"Writing modified content to {API_FILE}...")
    write_file(API_FILE, content)

    print("✓ Integration complete!")
    print("\nNext steps:")
    print("1. Review the changes in miniprogram_api.py")
    print("2. Update /api/chat/start endpoint to use database persistence")
    print("3. Update /api/chat/message endpoint to use database persistence")
    print("4. Update /api/chat/rate endpoint to use database persistence")
    print("5. Test the new endpoints")

if __name__ == "__main__":
    main()
