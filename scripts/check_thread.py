
import asyncio
import asyncpg
import os
from pathlib import Path

async def main():
    # Load .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

    db_url = os.environ.get("DATABASE_URL")
    print(f"Connecting to: {db_url}")
    
    try:
        conn = await asyncpg.connect(db_url)
        # Check specifically for "Thread"
        row = await conn.fetchrow("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'Thread'
        """)
        
        if row:
            print("Found table: Thread")
        else:
            print("Table 'Thread' NOT FOUND")
            # List what IS there
            rows = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            print("All tables:", [r['table_name'] for r in rows])
            
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
