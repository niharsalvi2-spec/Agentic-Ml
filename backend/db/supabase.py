from supabase import create_client, Client
from backend.core.config import settings

def get_supabase() -> Client:
    supabase_url: str = settings.SUPABASE_URL
    supabase_key: str = settings.SUPABASE_KEY
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment.")
    
    return create_client(supabase_url, supabase_key)

supabase = get_supabase()
