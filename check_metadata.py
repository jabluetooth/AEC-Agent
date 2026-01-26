
import asyncio
import asyncpg
import os
import json
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
        
        # Check Projects
        project_count = await conn.fetchval("SELECT COUNT(*) FROM projects")
        print(f"\nProjects found: {project_count}")
        
        if project_count > 0:
            project = await conn.fetchrow("SELECT * FROM projects LIMIT 1")
            print(f"Sample Project: {project['name']} (Source: {project['source']})")

        # Check Elements
        element_count = await conn.fetchval("SELECT COUNT(*) FROM elements")
        print(f"\nElements found: {element_count}")
        
        if element_count > 0:
            print("\nSample Element Metadata:")
            # Get an element that has properties
            element = await conn.fetchrow("SELECT entity_type, layer, properties FROM elements WHERE properties::text != '{}' LIMIT 1")
            if element:
                print(f"Type: {element['entity_type']}")
                print(f"Layer: {element['layer']}")
                props = json.loads(element['properties'])
                print(f"Properties: {json.dumps(props, indent=2)}")
            else:
                print("No elements with properties found (only empty records?).")
        else:
            print("\nNo elements found yet. Have you run the 'Sync Metadata' tool or opened a drawing?")

        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
